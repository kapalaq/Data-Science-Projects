import numpy as np
from scipy.signal import fftconvolve


class FastFourierTransform:

    def padding(self, arr, target):
        padded = np.pad(arr, (0, target - len(arr)), mode = "constant")
        return padded

    def transform(self, arr):
        n = len(arr)
        if n <= 1:
            return arr
        omega = np.exp(2 * np.pi * 1j / n)

        even, odd = arr[::2], arr[1::2]

        ye, yo = self.transform(even), self.transform(odd)

        y = np.zeros(n, dtype=complex)

        for i in range(n // 2):
            y[i] = ye[i] + (omega ** i) * yo[i]
            y[n // 2 + i] = ye[i] - (omega ** i) * yo[i]

        return y

    def inverse(self, arr):
        n = len(arr)
        if n <= 1:
            return arr
        omega = np.exp(-2 * np.pi * 1j / n)

        even, odd = arr[::2], arr[1::2]

        ye, yo = self.inverse(even), self.inverse(odd)

        y = np.zeros(n, dtype=complex)

        for i in range(n // 2):
            y[i] = ye[i] + (omega ** i) * yo[i]
            y[n // 2 + i] = ye[i] - (omega ** i) * yo[i]

        return y

    def convolve(self, a, b):
        c = len(a) + len(b) - 1
        n = 2 ** np.ceil(np.log2(c)).astype(int)

        a_pad, b_pad = self.padding(a, n), self.padding(b, n)

        a_tf, b_tf = self.transform(a_pad), self.transform(b_pad)

        ans = np.real(self.inverse(a_tf * b_tf))[:c]

        return ans / n


if __name__ == '__main__':
    a = np.random.random_integers(1, 10, size=15)
    b = np.random.random_integers(1, 10, size=2)
    fft = FastFourierTransform()
    hm = fft.convolve(a, b)
    scp = fftconvolve(a, b)
    print("\thandmade\tscipy")
    for i in range(min(len(hm), len(scp))):
        print("%d\t%.3f\t\t%.3f" % (i + 1, hm[i], scp[i]))
