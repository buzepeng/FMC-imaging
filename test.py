import numpy as np
from scipy import signal, fftpack

signal_path = "data/YJBL_5L64_0p5_100MHz_4000.csv"
TOF_path = "data/TOF_Data_40_40_0p04Plane.csv"
with open(signal_path, encoding='utf-8') as f:
    offLineFmcMat = np.loadtxt(f, delimiter=',')
with open(TOF_path, encoding='utf-8') as f:
    iTof = np.loadtxt(f, delimiter=',')
iTof = iTof.T

N = 2*4+1
WaveNum = 2048
element_freq = 5
sample_freq = 100
iWaveLength = 4000
low_cut = ((element_freq - 3)*2)/sample_freq
high_cut = ((element_freq+3)*2)/sample_freq
b, a = signal.butter(N, [low_cut, high_cut], 'bandpass')
FmcMatHilbert = np.empty((WaveNum, iWaveLength), dtype = np.complex)
for i in range(WaveNum):
    filterWave = signal.filtfilt(b, a, offLineFmcMat[i][2:])
    FmcMatHilbert[i] = fftpack.hilbert(filterWave)

NX, NZ = 1001, 1001
SignalNum = 2048
PixelNum = NX*NZ
MaxInTfmArray = 0

TfmArray = np.empty(NZ*NX, dtype = np.float32)
for i in range(PixelNum):
    real = 0
    imag = 0
    for s in range(SignalNum):
        tIndex, rIndex = int(offLineFmcMat[s][0]-1), int(offLineFmcMat[s][1]-1)
        trTofIndex = int(iTof[tIndex][i] + iTof[rIndex][i])
        real += FmcMatHilbert[s][trTofIndex-1].real
        imag += FmcMatHilbert[s][trTofIndex-1].imag
    TfmArray[i] = np.sqrt(np.power(real, 2)+np.power(imag, 2))
    MaxInTfmArray = max(MaxInTfmArray, TfmArray[i])

MindB = -30
TfmImage = np.empty((NZ, NX), dtype = np.float32)
for Z in range(NZ):
    for X in range(NX):
        value = TfmArray[Z*NX+X]
        dBvalue = max(20*np.log10(value/MaxInTfmArray), MindB)
        TfmImage[Z][X] = dBvalue

np.save('save_TfmImage', TfmImage)