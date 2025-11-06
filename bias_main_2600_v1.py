#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bias_main_2600_v1.py
- CLI/위저드 없이, 이 파일 상단 CONFIG만 수정해서 실행
- 실행: python bias_main_2600_v1.py
"""

from __future__ import annotations
import sys
import math
from typing import Dict, Any, List, Tuple

# 로컬 모듈
from bias_module_2600 import BiasController, segments_total_time, parse_segments  # parse_segments는 문자열 세그먼트 사용시 편의용

# ─────────────────────────────────────────────────────────────────────────────
# 🔧 CONFIG: 여기만 수정해서 씀
CONFIG: Dict[str, Any] = {
    "ip": "192.168.0.2",      # 계측기 IP
    "port": 5025,             # 계측기 포트

    # 실시간 플롯 표시 (True/False)
    "realtime_plot": True,

    # 채널별 설정
    "channels": {
        # ── Channel A ────────────────────────────────────────────────────────
        "a": {
            "enable": True,          # 사용 여부
            # 세그먼트: (시간[s], 전압[V]) 튜플 리스트 또는 "t,v; t,v; ..." 문자열
            "segments": [
                (0.50, 0.20),
                (0.50, 0.60),
            ],
            "cycles": 3,             # 주기 반복 횟수
            "sample_ms": 20,         # 샘플링 간격(ms)
            "compliance": 0.01,      # A
            "range_v": 10.0,         # V (예: 10 → ±10 V)
            "range_i": 0.1,          # A (예: 0.1 → 100 mA)
            "nplc": 0.01,            # 0.01 ~ 1.0 권장
            "transition": "step",    # "step" or "ramp"
            "ramp_ms": 5,            # ramp일 때만 사용
            "zero_ms": 1000,         # 시작/종료 0V 유지 시간(ms)
        },

        # ── Channel B ────────────────────────────────────────────────────────
        "b": {
            "enable": False,         # 필요 시 True로
            # 문자열로 적고 싶으면 → "0.5,0.2; 0.5,0.8; 1.0,0.0"
            "segments": [
                (0.50, 0.20),
                (0.50, 0.80),
                (1.00, 0.00),
            ],
            "cycles": 5,
            "sample_ms": 10,
            "compliance": 0.02,
            "range_v": 10.0,
            "range_i": 0.1,
            "nplc": 0.01,
            "transition": "ramp",
            "ramp_ms": 5,
            "zero_ms": 1000,
        },
    },
}
# ─────────────────────────────────────────────────────────────────────────────


# 내부 상수 (라인 주파수에 따라 PLC → ms 환산)
LINE_FREQ_HZ = 60.0
PLC_MS = 1000.0 / LINE_FREQ_HZ  # ≈ 16.667 ms @ 60Hz


def _ensure_segments(seg) -> List[Tuple[float, float]]:
    """
    CONFIG에서 세그먼트를 리스트/문자열 어느 쪽으로 주든 표준 리스트로 변환.
    """
    if isinstance(seg, str):
        return parse_segments(seg)
    return [(float(t), float(v)) for (t, v) in seg]


def _summarize_channel(ch: str, cfg: Dict[str, Any]) -> None:
    segs = _ensure_segments(cfg["segments"])
    total_t = segments_total_time(segs)
    integ_ms = float(cfg.get("nplc", 0.01)) * PLC_MS

    print(f"\n[SMU{ch.upper()}]")
    print(f"  enable        : {cfg.get('enable')}")
    print(f"  cycles        : {cfg.get('cycles')}")
    print(f"  sample_ms     : {cfg.get('sample_ms')} ms")
    print(f"  nplc          : {cfg.get('nplc')}  (integration ≈ {integ_ms:.2f} ms @ 60Hz)")
    print(f"  compliance    : {cfg.get('compliance')} A")
    print(f"  range_v       : {cfg.get('range_v')} V    (fixed; autorange OFF)")
    print(f"  range_i       : {cfg.get('range_i')} A    (fixed; autorange OFF)")
    print(f"  transition    : {cfg.get('transition')} (ramp_ms={cfg.get('ramp_ms')} ms)")
    print(f"  zero_ms       : {cfg.get('zero_ms')} ms")
    print(f"  segments (N={len(segs)}), total={total_t:.3f} s : {segs}")

    # 경고: 샘플 간격이 통합시간보다 짧은 경우
    if cfg.get("sample_ms", 20) < integ_ms:
        print(f"  [WARN] sample_ms({cfg.get('sample_ms')} ms) < NPLC integration({integ_ms:.2f} ms). "
              f"샘플 간격을 키우거나 NPLC를 낮추세요.", file=sys.stderr)


def main():
    ip = CONFIG["ip"]
    port = int(CONFIG.get("port", 5025))
    show_plot = bool(CONFIG.get("realtime_plot", True))

    # 채널 설정 수집
    channels_cfg = {}
    for ch in ("a", "b"):
        ch_cfg = CONFIG["channels"].get(ch, {})
        if not ch_cfg or not ch_cfg.get("enable", False):
            continue

        segs = _ensure_segments(ch_cfg["segments"])
        if not segs:
            print(f"[ERROR] 채널 {ch.upper()} 세그먼트가 비었습니다.", file=sys.stderr)
            sys.exit(2)

        channels_cfg[ch] = dict(
            segments=segs,
            cycles=int(ch_cfg.get("cycles", 1)),
            sample_ms=int(ch_cfg.get("sample_ms", 20)),
            compliance=float(ch_cfg["compliance"]),
            range_v=float(ch_cfg["range_v"]),
            range_i=float(ch_cfg["range_i"]),
            nplc=float(ch_cfg.get("nplc", 0.01)),
            transition=ch_cfg.get("transition", "step"),
            ramp_ms=int(ch_cfg.get("ramp_ms", 5)),
            zero_ms=int(ch_cfg.get("zero_ms", 1000)),
        )

    if not channels_cfg:
        print("[INFO] 활성화된 채널이 없습니다. CONFIG에서 'enable'을 True로 설정하세요.")
        sys.exit(0)

    # 실행 요약 프린트
    print("=== Keithley 2600 Time-Segmented Bias ===")
    print(f"IP={ip}, Port={port}, RealtimePlot={show_plot}")
    for ch in channels_cfg.keys():
        _summarize_channel(ch, {**CONFIG["channels"][ch], "segments": channels_cfg[ch]["segments"]})

    # 빌드 & 실행
    bc = BiasController(ip=ip, port=port, channels_cfg=channels_cfg)
    bc.build()
    bc.run(realtime_plot=show_plot)


if __name__ == "__main__":
    main()
