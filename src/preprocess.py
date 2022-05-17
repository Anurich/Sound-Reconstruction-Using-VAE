import pickle
from posixpath import split
from torch import sign
import config
import librosa
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split

class preprocessing:
    def __init__(self, filepath, audiopath) -> None:
        self.filepath = filepath
        self.df = pd.read_csv(self.filepath)[["slice_file_name", "fold", "classID", "class"]]
        self.df["path"] = "fold"
        self.df["path"] = audiopath+"/"+self.df["path"] + self.df["fold"].astype(str)+"/"+self.df["slice_file_name"]
    
    def _savepickle(self, dictionary, filepath):
        with open(filepath, "wb") as fp:
            pickle.dump(dictionary, fp)

    def _createindexes(self, df):
        index2word = {rows["classID"]:rows["class"] for idx, rows in df.iterrows()}
        word2index = {value:key for key,value in index2word.items() }
        self._savepickle(index2word, config.SAVE_INDEX2WORD)
        self._savepickle(word2index, config.SAVE_WORD2INDEX)
        return index2word, word2index

    def _normalize(self, stft, idx):
        normalized_stft = config.SCALER.fit_transform(stft)
        return normalized_stft

    def _padding(self, signal):
        # we pad the value if it is less then current Num samples 
        # else we truncate 
        if signal.shape[0] < config.NUM_SAMPLES:
            left_signal = config.NUM_SAMPLES  - signal.shape[0]
            signal = np.pad(signal,(0,left_signal), constant_values=0)
        elif signal.shape[0] > config.NUM_SAMPLES:
            signal = signal[:config.NUM_SAMPLES]
        return signal

    def _procossedaudio(self, df, word2index):
        x = []
        y = []
        for idx, rows in tqdm(df.iterrows()):
            label = rows["class"]
            signal, sr = librosa.load(rows["path"])
            signal = self._padding(signal)
            stft = librosa.stft(signal, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH)
            stft = np.abs(stft[:-1])
            signal = librosa.amplitude_to_db(stft)
            normalized_stft = self._normalize(signal, idx)
            x.append(normalized_stft)
            y.append(word2index[label])
        return {"features": x, "label": y}
    
    def _preprocessingMEL(self, df, word2index):
        x = []
        y = []
        for idx, rows in tqdm(df.iterrows()):
            label = rows["class"]
            signal, sr = librosa.load(rows["path"])
            signal = self._padding(signal)
            signal = librosa.feature.melspectrogram(y = signal, sr = sr, n_fft=config.MEL_N_FFT, hop_length=config.MEL_HOP_LENGTH,\
                n_mels =128)
            signal = librosa.power_to_db(signal)
            signal = self._normalize(signal,idx)
            x.append(signal)
            y.append(word2index[label])
        return {"features": x, "label": y}

    def _processingMFCC(self, df, word2index):
        x = []
        y = []
        for idx, rows in tqdm(df.iterrows()):
            label = rows["class"]
            signal, sr = librosa.load(rows["path"])
            signal = self._padding(signal)
            signal = librosa.feature.mfcc(y = signal, sr = sr, n_mfcc= config.N_MFCC , n_fft = config.N_FFT, hop_length =config.HOP_LENGTH)
            signal = signal.T
            signal_order1 = librosa.feature.delta(signal, order=1)
            signal_order2 = librosa.feature.delta(signal, order=2)
            #signal = np.mean(signal, axis=0, keepdims=True)
            #signal = librosa.amplitude_to_db(signal)
            #signal = librosa.power_to_db(signal)
            signal = self._normalize(signal_order2,idx)
            x.append(signal)
            y.append(word2index[label])
        return {"features": x, "label": y}



    def __call__(self):
        _, word2index = self._createindexes(self.df)
        feature_label = self._processingMFCC(self.df, word2index)
        self._savepickle(feature_label, config.SAVE_FEATURES_LABELS)

class splitting:
    def __init__(self, filepath) -> None:
        self.filepath = filepath
        self.data = self._readpickle()

        self.x = np.array(self.data["features"])
        self.y = np.array(self.data["label"])


    def _readpickle(self):
        return pickle.load(open(self.filepath, "rb"))

    def _writepickle(self, data, path):
        with open(path, "wb") as fp:
            pickle.dump(data, fp)

    def __call__(self) -> None:
        #let's split into test and train 
        trainX, testX, trainY, testY = train_test_split(self.x, self.y, test_size= 0.3, stratify=self.y)
        self._writepickle([trainX, trainY], config.SAVE_FEATURES_TRAIN)
        self._writepickle([testX, testY], config.SAVE_FEATURES_TEST)


prep = preprocessing(config.CSV_FILE_PATH, config.AUDIO_FILES_PATH)
prep()
spt = splitting(config.SAVE_FEATURES_LABELS)
spt()



