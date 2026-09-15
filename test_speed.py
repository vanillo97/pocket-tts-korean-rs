"""synthesize.py apply_speed 자체 점검: python3 test_speed.py"""
import importlib.util
import numpy as np

spec = importlib.util.spec_from_file_location("s", "synthesize.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

# 유성음 유사 신호 (F0 120Hz + 배음), 2초 24kHz
sr, n = 24000, 48000
t = np.arange(n) / sr
a = (np.sin(2 * np.pi * 120 * t) + 0.5 * np.sin(2 * np.pi * 240 * t)).astype(np.float32)


def f0(x):
    x = x - x.mean()
    c = np.correlate(x, x, "full")[len(x) - 1:]
    lo, hi = sr // 400, sr // 60
    return sr / (np.argmax(c[lo:hi]) + lo)


for speed in (1.0, 0.5, 0.75, 1.5, 2.0):
    out = m.apply_speed(a, speed)
    want = round(n / speed)
    assert out.size == want, f"speed {speed}: len {out.size}, want {want}"
    got = f0(out[sr // 2: sr // 2 + 4096])
    assert abs(got - 120) < 6, f"speed {speed}: F0 {got:.1f}Hz drifted (피치 보존 실패)"

assert m.apply_speed(np.zeros(500, np.float32), 2.0).size == 250  # 짧은 입력 폴백
for bad in (0.1, 3.0):
    try:
        m.apply_speed(a, bad)
        raise AssertionError(f"speed {bad} should be rejected")
    except ValueError:
        pass
print("ok")
