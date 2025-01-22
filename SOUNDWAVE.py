import matplotlib.pyplot as plt
from scipy.io import wavfile as wav
from scipy.fftpack import fft,ifft
import numpy as np

data=wav.read('C:\')
audio=data(1)
plt.plot(audio[0:50])
plt.plot(audio)

fft_abs=fft(data[1])
plt.figure[1]
plt.plot(data[1],np.abs(fft_abs))
fft_in=ifft(data[1])
plt.figure(2)
plt.plot(data[1],(fft_in))
plt.show()