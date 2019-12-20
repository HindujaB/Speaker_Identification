import numpy as np
from sklearn import preprocessing
from python_speech_features import mfcc
from python_speech_features import delta


class FeaturesExtractor:
    def __init__(self):
        pass

    def extract_features(self, audio_array, sample_rate):
        """
        Extract voice features including the Mel Frequency Cepstral Coefficient (MFCC)
        from an audio using the python_speech_features module, performs Cepstral Mean
        Normalization (CMS) and combine it with MFCC deltas and the MFCC double
        deltas.

        Args:
            audio_path (str) : path to wave file without silent moments.

        Returns:
            (array) : Extracted features matrix.
        """
        mfcc_feature = mfcc(  # The audio signal from which to compute features.
            audio_array,
            # The samplerate of the signal we are working with.
            sample_rate,
            # The length of the analysis window in seconds.
            # Default is 0.025s (25 milliseconds)
            winlen=0.025,
            # The step between successive windows in seconds.
            # Default is 0.01s (10 milliseconds)
            winstep=0.01,
            # The number of cepstrum to return.
            # Default 13.
            numcep=13,
            # The number of filters in the filterbank.
            # Default is 26.
            nfilt=26,
            # The FFT size. Default is 512.
            nfft=512,
            # If true, the zeroth cepstral coefficient is replaced
            # with the log of the total frame energy.
            appendEnergy=True)

        # Normalizing using mean-variance normalizing
        mfcc_feature = preprocessing.scale(mfcc_feature)
        # print('mfcc shape : ', mfcc_feature.shape)
        deltas = delta(mfcc_feature, 2)
        # print('delta shape : ', deltas.shape)
        double_deltas = delta(deltas, 2)
        # print('double delta shape : ', double_deltas.shape)

        combined = np.hstack((mfcc_feature, deltas, double_deltas))
        print("Shape of the training features : ", combined.shape)
        return combined

    def accelerated_get_features_vector(self, input_wave_file, audio_array, sample_rate):
        """
        Get voice features from an input wave file faster.

        Args:
            input_wave_file (str) : Path to input wave file.
            audio_array       (ndarray) : Array representing the wave data.
            sample_rate      (int) : Rate of the audio.

        Returns:
            (array) with the voice features if the extraction was successful else [].
        """
        # extract features
        try:
            return self.extract_features(audio_array, sample_rate)

        except:
            print("Cannot extract features from", input_wave_file.split('/')[-1])
            return np.array([])
