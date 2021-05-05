import numpy as np
from scipy.stats import entropy
from scipy.signal import find_peaks
from math import sqrt

class FeatureExtractor():
    def __init__(self, debug=True):
        self.debug = debug

    def _compute_mean_features(self, window):
        """
        Category 1 - Statistical Feature
        Computes the mean x, y and z acceleration over the given window. 
        """
        return np.mean(window, axis=0)

    def _compute_variance_features(self, window):
        """
        Category 1 - Statistical Feature
        Computes the variance of x, y, and z over the given window
        """
        return np.var(window, axis=0)

    def _compute_dominant_freq(self, xSignal, ySignal, zSignal, magnitudeSignal):
        """
        Category 2 - FFT Feature
        Fast Fourier Transform - computes real-value Discrete Fourier Transform and decomposes signal into finite number of sinusoidal components
        - run FFT to split into all the frequencies
        - return highest/dominant frequency for each split list
        """
        dominantFrequencyX = max(np.fft.rfft(xSignal, axis=0).astype(float))
        dominantFrequencyY = max(np.fft.rfft(ySignal, axis=0).astype(float))
        dominantFrequencyZ = max(np.fft.rfft(zSignal, axis=0).astype(float))
        dominantFrequencyMagnitude = max(
            np.fft.rfft(magnitudeSignal, axis=0).astype(float))
        return np.array([dominantFrequencyX, dominantFrequencyY, dominantFrequencyZ, dominantFrequencyMagnitude])

    def _compute_entropy(self, xSignal, ySignal, zSignal, magnitudeSignal, base=None):
        """
        Category 3 - Other Features
        Entropy - measure of how pure/homogenous a group is
        - count unique values inside signal
        - return entropy calculation given count of unique values
        """
        valueX, countsX = np.unique(xSignal, return_counts=True)
        valueY, countsY = np.unique(ySignal, return_counts=True)
        valueZ, countsZ = np.unique(zSignal, return_counts=True)
        valueMagnitude, countsMagnitude = np.unique(
            magnitudeSignal, return_counts=True)
        return np.array([entropy(countsX, base=base), entropy(countsY, base=base), entropy(countsZ, base=base), entropy(countsMagnitude, base=base)])

    def _compute_peak_count(self, xSignal, ySignal, zSignal, magnitudeSignal):
        """
        Category 4 - Peak Features
        find number of peaks in window
        """
        return np.array([len(find_peaks(xSignal)), len(find_peaks(ySignal)), len(find_peaks(zSignal)), len(find_peaks(magnitudeSignal))], dtype="object")  # prob array so requires dtype as object (tho find_peaks returns float)

    def extract_features(self, window, debug=False):
        """
        Here is where you will extract your features from the data in 
        the given window.

        Make sure that x is a vector of length d matrix, where d is the number of features.
        '''
        feature_names = []
        # Mean
        feature_names.append("x_mean")
        feature_names.append("y_mean")
        feature_names.append("z_mean")
        feature_names.append("magnitude_mean")
        # Maximum
        feature_names.append("x_variance")
        feature_names.append("y_variance")
        feature_names.append("z_variance")
        feature_names.append("magnitude_variance")
        # FFT Features
        feature_names.append("x_dominant_frequency")
        feature_names.append("y_dominant_frequency")
        feature_names.append("z_dominant_frequency")
        feature_names.append("magnitude_dominant_frequency")
        # Other Features
        feature_names.append("x_entropy")
        feature_names.append("y_entropy")
        feature_names.append("z_entropy")
        feature_names.append("magnitude_entropy")
        # Peak Features
        feature_names.append("x_peak_count")
        feature_names.append("y_peak_count")
        feature_names.append("z_peak_count")
        feature_names.append("magnitude_peak_count")
        '''
        """
        print(window)
        x = []

        xSignal = []
        ySignal = []
        zSignal = []
        magnitudeSignal = []
        for i in range(len(window)):
            xSignal.append(window[i][0])
            ySignal.append(window[i][1])
            zSignal.append(window[i][2])
            magnitudeSignal.append(window[i][3])
            # magnitudeSignal.append(sqrt(window[i][0] ** 2 + window[i][1] ** 2 + window[i][2] ** 2))

        x.append(self._compute_mean_features(window))  # appends 3 values (x, y, z)
        x.append(self._compute_variance_features(window))
        x.append(self._compute_dominant_freq(xSignal, ySignal,
                zSignal, magnitudeSignal))  # 2D Array
        x.append(self._compute_entropy(xSignal, ySignal, zSignal, magnitudeSignal))
        x.append(self._compute_peak_count(xSignal, ySignal, zSignal, magnitudeSignal))

        # convert the list of features to a single 1-dimensional vector
        feature_vector = np.array(x, dtype="object").flatten()
        
        # feature_vector = np.concatenate(x, axis=0) # convert the list of features to a single 1-dimensional vector
        return feature_vector