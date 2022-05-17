import os
from sklearn.preprocessing import MinMaxScaler
import pickle
import sys


ABS_PATH = os.path.abspath(".")
MODEL_PATH = sys.path.insert(0, os.path.join(ABS_PATH, "models"))
AUDIO_FILES_PATH  = os.path.join(ABS_PATH,"data", "audio")
CSV_FILE_PATH     = os.path.join(ABS_PATH,"data", "UrbanSound8k.csv")
SAVE_MODEL = os.path.join(ABS_PATH, "models", "model.pth")
SAVE_WORD2INDEX  =  os.path.join(ABS_PATH, "data", "save_indexes", "word2index.pickle")
SAVE_INDEX2WORD  =  os.path.join(ABS_PATH, "data", "save_indexes", "index2word.pickle")
SAVE_FEATURES_LABELS = os.path.join(ABS_PATH, "data","save_indexes", "features_labels.pickle")
SAVE_FEATURES_TRAIN  = os.path.join(ABS_PATH, "data", "save_indexes", "features_train.pickle")
SAVE_FEATURES_TEST  = os.path.join(ABS_PATH, "data", "save_indexes", "features_test.pickle")
SAVE_PRED_SOUND = os.path.join(ABS_PATH, "data", "pred_sound")
SAVE_PRED_SOUND_ORIGINAL = os.path.join(ABS_PATH, "data", "pred_sound")
N_FFT = 1024
MEL_N_FFT = 1024
HOP_LENGTH = 512
MEL_HOP_LENGTH = 512
NUM_SAMPLES = 22050
BATCH_SIZE  = 8
LATENT_SPACE = 2
LR = 0.0001
ITERATION = 1000
N_MFCC = 13

SCALER = MinMaxScaler()

def loadpickle(filepath):
    return pickle.load(open(filepath, "rb"))

def savepickle(dictionary, filepath):
    with open(filepath, "wb") as fp:
        pickle.dump(dictionary, fp)

