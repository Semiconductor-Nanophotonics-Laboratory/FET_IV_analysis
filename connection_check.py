from keithley_controller import Keithley2600

ip_address = "192.168.100.2"  # 장비 IP

k = Keithley2600(ip_address)
k.connect()

print("=== SMUA 테스트 ===")
resp_a = k.query("print(smua.measure.i())")
print("SMUA Current:", resp_a)

print("=== SMUB 테스트 ===")
resp_b = k.query("print(smub.measure.i())")
print("SMUB Current:", resp_b)

k.close()