"""make_voice.py cap_length 자체 점검: python3 test_make_voice.py"""
import importlib.util
import numpy as np

spec = importlib.util.spec_from_file_location("mv", "make_voice.py")
mv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mv)
SR = mv.SAMPLE_RATE


def speech(dur_s, amp=0.5):
    t = np.arange(int(dur_s * SR)) / SR
    return (np.sin(2 * np.pi * 150 * t) * amp).astype(np.float32)


def silence(dur_s):
    return np.zeros(int(dur_s * SR), dtype=np.float32)


# 말 2s / 쉼 0.5s / 말 2s / 쉼 0.5s / 말 2s  (총 7s, 발화 끝 = 2.0s, 4.5s, 7.0s)
a = np.concatenate([speech(2), silence(0.5), speech(2), silence(0.5), speech(2)])
assert abs(len(a) / SR - 7.0) < 0.01

# 상한이 입력보다 길면 그대로
assert len(mv.cap_length(a, 10.0)) == len(a)

# 5초 상한 → 4.5s 발화 경계에서 끊긴다 (5.0에서 자르면 단어 중간)
cut = len(mv.cap_length(a, 5.0)) / SR
assert 4.4 < cut <= 5.0, f"5s 상한: {cut:.2f}s, 4.5s 경계 기대"

# 3초 상한 → 2.0s 경계는 3.0*0.7=2.1 미만이라 버리는 양이 크다 → 3.0s 하드컷
cut = len(mv.cap_length(a, 3.0)) / SR
assert abs(cut - 3.0) < 0.02, f"3s 상한: {cut:.2f}s, 하드컷 기대"

# 경계가 아예 없는 연속 발화는 하드컷
cut = len(mv.cap_length(speech(7), 4.0)) / SR
assert abs(cut - 4.0) < 0.02, f"연속 발화: {cut:.2f}s"

# 잘라낸 결과가 원본의 접두사여야 한다 (뒤에서 자르지 않음)
out = mv.cap_length(a, 5.0)
assert np.array_equal(out, a[:len(out)])

print("ok")
