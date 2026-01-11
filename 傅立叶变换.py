import math
import cmath
import matplotlib.pyplot as plt

def dft(signal):
    """离散傅立叶变换：输入为实数信号列表，输出为复数频谱列表"""
    N = len(signal)
    spectrum = []
    for k in range(N):
        real = 0.0
        imag = 0.0
        for n in range(N):
            angle = 2 * math.pi * k * n / N
            real += signal[n] * math.cos(-angle)
            imag += signal[n] * math.sin(-angle)
        spectrum.append(complex(real, imag))
    return spectrum

def magnitude_spectrum(spectrum):
    """计算频谱的模值"""
    return [abs(freq) for freq in spectrum]

def magnitude_spectrum(spectrum):
    N = len(spectrum)
    return [abs(freq) / N for freq in spectrum]
# 示例信号：采样一段含两个频率的波形（5Hz + 20Hz）
sampling_rate = 100  # Hz
duration = 1.0       # 秒
N = int(sampling_rate * duration)
t = [i / sampling_rate for i in range(N)]
signal = [math.sin(2 * math.pi * 5 * ti) + 0.5 * math.sin(2 * math.pi * 20 * ti) for ti in t]

# 计算 DFT
spectrum = dft(signal)
magnitudes = magnitude_spectrum(spectrum)

# 频率坐标
freqs = [i * sampling_rate / N for i in range(N)]

# 画图
plt.figure(figsize=(10, 4))
plt.stem(freqs[:N // 2], magnitudes[:N // 2])  # 删除 use_line_collection
plt.title("Magnitude Spectrum (DFT using math only)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid(True)
plt.tight_layout()
plt.show()

'''
# ---- 时域波形图 ----
plt.figure(figsize=(10, 3))
plt.plot(t, signal, label='Combined Signal')
plt.title("Time Domain Signal (5Hz + 20Hz)")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()
plt.show()
'''
'''
# 分别构建单独的波形
signal_5hz = [math.sin(2 * math.pi * 5 * ti) for ti in t]
signal_20hz = [0.5 * math.sin(2 * math.pi * 20 * ti) for ti in t]

# 画图对比：单独两个波形 + 叠加波形
plt.figure(figsize=(12, 6))

plt.subplot(3, 1, 1)
plt.plot(t, signal_5hz, color='blue')
plt.title("5 Hz Sine Wave")
plt.ylabel("Amplitude")
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(t, signal_20hz, color='orange')
plt.title("20 Hz Sine Wave (0.5 Amplitude)")
plt.ylabel("Amplitude")
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(t, signal, color='green')
plt.title("Combined Signal: 5 Hz + 0.5 * 20 Hz")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)

plt.tight_layout()
plt.show()
'''
# 分别构建单独波形
signal_5hz = [math.sin(2 * math.pi * 5 * ti) for ti in t]
signal_20hz = [0.5 * math.sin(2 * math.pi * 20 * ti) for ti in t]

# 同一张图上画三条线
plt.figure(figsize=(12, 4))
plt.plot(t, signal_5hz, label="5 Hz", color="blue")
plt.plot(t, signal_20hz, label="20 Hz (×0.5)", color="orange")
plt.plot(t, signal, label="Combined", color="green")

plt.title("Time Domain Signals: 5 Hz, 20 Hz and Combined")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()