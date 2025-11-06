#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2600_bias_module.py
Keithley 2600 (SMUA/SMUB) time-segmented voltage bias driver + real-time I/V streaming.

핵심 기능
- 채널 독립 동작(SMUA/SMUB)
- 시간-세그먼트 기반 전압 인가 (step/ramp 전이)
- 전류 측정 + 전압 측정 동시 수행
- 샘플링 간격 고정, 컴플라이언스, 고정 레인지(오토레인지 OFF → 지연 방지)
- 시작/종료 0V 유지
- NPLC에 따른 최소 샘플 간격 경고
- 안전 종료(Ctrl+C) 시 0V로 램프다운 후 출력 OFF

필수 의존:
- keithley_controller.py (동일 폴더)
"""
from __future__ import annotations
import time
import threading
from collections import deque
import sys
from typing import List, Tuple, Optional, Dict

import numpy as np

# 통신 래퍼 (사용자 제공)
from keithley_controller import Keithley2600

LINE_FREQ_HZ = 60.0             # KR: 60 Hz
PLC_MS = 1000.0 / LINE_FREQ_HZ  # ≈ 16.667 ms


def parse_segments(seg_str: str) -> List[Tuple[float, float]]:
    """
    "t1,v1; t2,v2; ..." -> [(t1, v1), (t2, v2), ...]
    시간: 초(s), 전압: V
    공백 허용. 빈 문자열 → [].
    """
    if not seg_str:
        return []
    pairs: List[Tuple[float, float]] = []
    for p in seg_str.split(';'):
        p = p.strip()
        if not p:
            continue
        t_str, v_str = p.split(',')
        t = float(t_str.strip())
        v = float(v_str.strip())
        assert t >= 0, "세그먼트 시간은 0 이상이어야 합니다."
        pairs.append((t, v))
    return pairs


def segments_total_time(segments: List[Tuple[float, float]]) -> float:
    return float(sum(t for t, _ in segments))


def _tsp_init_channel(ch: str, compliance: float, range_v: float, range_i: float, nplc: float) -> List[str]:
    smu = f"smu{ch}"
    return [
        f"{smu}.reset()",
        f"{smu}.source.func = {smu}.OUTPUT_DCVOLTS",
        f"{smu}.source.autorangev = {smu}.AUTORANGE_OFF",
        f"{smu}.source.rangev = {range_v}",

        f"{smu}.sense = {smu}.SENSE_LOCAL",  # 2-wire (원하면 REMOTE로 4-wire)
        f"{smu}.measure.autorangei = {smu}.AUTORANGE_OFF",
        f"{smu}.measure.rangei = {range_i}",

        f"{smu}.source.limiti = {compliance}",
        f"{smu}.measure.nplc = {nplc}",

        f"{smu}.source.levelv = 0",
        f"{smu}.source.output = {smu}.OUTPUT_ON"
    ]


def _tsp_set_v(ch: str, v: float) -> List[str]:
    smu = f"smu{ch}"
    return [f"{smu}.source.levelv = {v}"]


def _tsp_measure_iv(ch: str) -> str:
    smu = f"smu{ch}"
    # I, V를 한 줄로 프린트하여 왕복 통신 최소화
    return f'print(string.format("%e,%e", {smu}.measure.i(), {smu}.measure.v()))'


def _tsp_output_off(ch: str) -> List[str]:
    smu = f"smu{ch}"
    return [f"{smu}.source.levelv = 0", f"{smu}.source.output = {smu}.OUTPUT_OFF"]


class ChannelRunner(threading.Thread):
    """
    하나의 SMU 채널(SMUA 또는 SMUB)을 구동하는 스레드.
    """
    def __init__(
        self,
        ip: str,
        port: int,
        ch: str,  # 'a' or 'b'
        segments: List[Tuple[float, float]],
        cycles: int,
        sample_ms: int,
        compliance: float,
        range_v: float,
        range_i: float,
        nplc: float,
        transition: str = "step",
        ramp_ms: int = 5,
        zero_ms: int = 1000,
        plot_buf: int = 20000,
    ):
        super().__init__(daemon=True)
        self.ip = ip
        self.port = port
        self.ch = ch
        self.segments = segments
        self.cycles = cycles
        self.sample_ms = sample_ms
        self.compliance = compliance
        self.range_v = range_v
        self.range_i = range_i
        self.nplc = nplc
        self.transition = transition
        self.ramp_ms = ramp_ms
        self.zero_ms = zero_ms

        self.ts = deque(maxlen=plot_buf)
        self.Vs = deque(maxlen=plot_buf)
        self.Is = deque(maxlen=plot_buf)

        self._stop = threading.Event()
        self._exc: Optional[BaseException] = None

    def stop(self):
        self._stop.set()

    def get_exception(self) -> Optional[BaseException]:
        return self._exc

    # 내부 유틸
    def _read_and_append(self, k: Keithley2600):
        resp = k.query(_tsp_measure_iv(self.ch))
        try:
            i_str, v_str = resp.split(',')
            I = float(i_str)
            V = float(v_str)
        except Exception:
            # 비정상 응답 방어
            I, V = np.nan, np.nan
        self.ts.append(time.time())
        self.Is.append(I)
        self.Vs.append(V)

    def _read_v_only(self, k: Keithley2600) -> float:
        try:
            v = float(k.query(f'print(smu{self.ch}.measure.v())').strip())
            return v
        except Exception:
            return 0.0

    def _ramp_to(self, k: Keithley2600, target_v: float):
        """
        sample_ms 간격을 기준으로 ramp_ms 동안 선형 램프 수행
        (오버슈트 방지 목적, 너무 빈번한 통신은 피함)
        """
        cur_v = self._read_v_only(k)
        if self.ramp_ms <= 0:
            for c in _tsp_set_v(self.ch, target_v):
                k.send(c)
            return
        steps = max(1, int(self.ramp_ms / max(1, self.sample_ms)))
        for i in range(1, steps + 1):
            if self._stop.is_set():
                break
            v = cur_v + (target_v - cur_v) * (i / steps)
            for c in _tsp_set_v(self.ch, v):
                k.send(c)
            time.sleep(self.sample_ms / 1000.0)

    def run(self):
        try:
            self._run_impl()
        except BaseException as e:
            self._exc = e
            self._stop.set()

    def _run_impl(self):
        k = Keithley2600(self.ip, self.port, timeout=5)
        k.connect()

        # 초기화 및 출력 ON
        for c in _tsp_init_channel(self.ch, self.compliance, self.range_v, self.range_i, self.nplc):
            k.send(c)

        # 시작 0V 유지
        t0 = time.time()
        while (time.time() - t0) * 1000.0 < self.zero_ms and not self._stop.is_set():
            self._read_and_append(k)
            time.sleep(0.05)

        # NPLC vs sample 간격 체크
        integ_ms = self.nplc * PLC_MS
        if self.sample_ms < integ_ms:
            print(
                f"[WARN][SMU{self.ch.upper()}] sample_ms({self.sample_ms}) < NPLC integration({integ_ms:.2f} ms).",
                file=sys.stderr
            )

        # 메인 루프: cycles × segments
        for cyc in range(int(self.cycles)):
            if self._stop.is_set():
                break
            for seg_t, seg_v in self.segments:
                if self._stop.is_set():
                    break
                # 전이
                if self.transition == "ramp":
                    self._ramp_to(k, seg_v)
                else:
                    for c in _tsp_set_v(self.ch, seg_v):
                        k.send(c)

                # dwell
                seg_start = time.time()
                while (time.time() - seg_start) < seg_t:
                    if self._stop.is_set():
                        break
                    self._read_and_append(k)
                    time.sleep(self.sample_ms / 1000.0)

        # 종료 0V 유지
        for c in _tsp_set_v(self.ch, 0.0):
            k.send(c)
        t1 = time.time()
        while (time.time() - t1) * 1000.0 < self.zero_ms and not self._stop.is_set():
            self._read_and_append(k)
            time.sleep(0.05)

        # 출력 OFF
        for c in _tsp_output_off(self.ch):
            k.send(c)
        k.close()


class BiasController:
    """
    두 채널을 함께 관리(선택적으로 하나만).
    외부(메인)에서는 이 컨트롤러에 파라미터를 넘기고 run()을 호출.
    """
    def __init__(
        self,
        ip: str,
        port: int,
        channels_cfg: Dict[str, Dict],  # {'a': {...}, 'b': {...}}
    ):
        self.ip = ip
        self.port = port
        self.channels_cfg = channels_cfg
        self.runners: List[ChannelRunner] = []

    def build(self):
        self.runners.clear()
        for ch, cfg in self.channels_cfg.items():
            segs = cfg["segments"]
            assert len(segs) > 0, f"채널 {ch.upper()} 세그먼트가 비어 있습니다."
            r = ChannelRunner(
                ip=self.ip,
                port=self.port,
                ch=ch,
                segments=segs,
                cycles=int(cfg.get("cycles", 1)),
                sample_ms=int(cfg.get("sample_ms", 20)),
                compliance=float(cfg["compliance"]),
                range_v=float(cfg["range_v"]),
                range_i=float(cfg["range_i"]),
                nplc=float(cfg.get("nplc", 0.01)),
                transition=cfg.get("transition", "step"),
                ramp_ms=int(cfg.get("ramp_ms", 5)),
                zero_ms=int(cfg.get("zero_ms", 1000)),
            )
            self.runners.append(r)

    def run(self, realtime_plot: bool = True):
        import matplotlib.pyplot as plt

        # 시작
        for r in self.runners:
            r.start()

        # 플롯 없음(헤드리스)
        if not realtime_plot:
            try:
                while any(r.is_alive() for r in self.runners):
                    time.sleep(0.1)
                for r in self.runners:
                    if r.get_exception():
                        raise r.get_exception()
            finally:
                for r in self.runners:
                    r.join(timeout=1.0)
            return

        # 실시간 플롯
        plt.ion()
        fig, axes = plt.subplots(len(self.runners), 1, figsize=(9, 4 * max(1, len(self.runners))), sharex=True)
        if len(self.runners) == 1:
            axes = [axes]

        linesV, linesI = [], []
        for ax, r in zip(axes, self.runners):
            ax.set_title(f"SMU{r.ch.upper()}  V(t) & I(t)")
            ax.set_xlabel("Time (s, relative)")
            ax.grid(True, alpha=0.3)
            lv, = ax.plot([], [], label="V [V]")
            li, = ax.plot([], [], label="I [A]")
            ax.legend(loc="best")
            linesV.append(lv)
            linesI.append(li)

        def update():
            for lv, li, r, ax in zip(linesV, linesI, self.runners, axes):
                ts = np.asarray(r.ts, float)
                if ts.size == 0:
                    continue
                t_rel = ts - ts[0]
                Vs = np.asarray(r.Vs, float)
                Is = np.asarray(r.Is, float)
                lv.set_data(t_rel, Vs)
                li.set_data(t_rel, Is)
                ax.relim()
                ax.autoscale_view()

            fig.canvas.draw()
            fig.canvas.flush_events()

        try:
            while any(r.is_alive() for r in self.runners):
                # 예외 전달
                for r in self.runners:
                    e = r.get_exception()
                    if e is not None:
                        raise e
                update()
                # 플롯 주기 = max(50ms, sample_ms)
                sp = max(0.05, min([r.sample_ms for r in self.runners]) / 1000.0)
                time.sleep(sp)
        except KeyboardInterrupt:
            print("\n[ABORT] 사용자 인터럽트: 0V로 램프다운 후 종료합니다.", file=sys.stderr)
            for r in self.runners:
                r.stop()
        finally:
            for r in self.runners:
                r.join(timeout=2.0)
            update()
            try:
                import matplotlib.pyplot as plt
                plt.ioff()
                plt.show()
            except Exception:
                pass
