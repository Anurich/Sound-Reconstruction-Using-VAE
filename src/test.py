# plotting 
from cProfile import label
import librosa
import soundfile as sf
import pandas as pd
import numpy as np
import os
from unicodedata import name
import config
from playsound import playsound
from vae import VAE
import torch
import matplotlib.pyplot  as plt


def plot():
    checkpoint = torch.load(config.SAVE_MODEL)
    test_loss = checkpoint["test_loss"]
    train_loss = checkpoint["train_loss"]

    plt.plot(test_loss, label="test_loss")
    plt.plot(train_loss, label="train_loss")
    plt.title("Loss")
    plt.xlabel("steps")
    plt.ylabel("loss")
    plt.legend()

    plt.savefig("train_test_loss.jpg")

# taken from https://github.com/musikalkemist/generating-sound-with-neural-networks/blob/4e71d22683edb9bd56aa46de3f022f4e1dec1cf1/14%20Sound%20generation%20with%20VAE/code/generate.py
def save_signals(signals, name,save_dir, sample_rate=22050):
    for i, signal in enumerate(signals):
        save_path = os.path.join(save_dir, name+"_"+str(i) + ".wav")
        sf.write(save_path, signal, sample_rate)


def padding(signal):
        # we pad the value if it is less then current Num samples 
        # else we truncate 
        if signal.shape[0] < config.NUM_SAMPLES:
            left_signal = config.NUM_SAMPLES  - signal.shape[0]
            signal = np.pad(signal,(0,left_signal), constant_values=0)
        elif signal.shape[0] > config.NUM_SAMPLES:
            signal = signal[:config.NUM_SAMPLES]
        return signal

def tosignal(output):
    denorm = config.SCALER.inverse_transform(output)
    denorm = librosa.feature.delta(denorm, order=1)
    denorm = denorm.T
    signal = librosa.feature.inverse.mfcc_to_mel(denorm)
    signal = librosa.feature.inverse.mel_to_audio(denorm)
    return signal

def test_results():
    vae = VAE(config.LATENT_SPACE)
    checkpoint = torch.load(config.SAVE_MODEL)
    vae.load_state_dict(checkpoint["model_state_dict"])
    vae.eval()
    df = pd.read_csv(config.CSV_FILE_PATH)
    path = os.path.join(config.AUDIO_FILES_PATH,"fold"+str(df["fold"].iloc[550])+"/"+df["slice_file_name"].iloc[550])
    playsound(path)
    orgsignal,sr = librosa.load(path)
    signal = padding(orgsignal)
    signal = librosa.feature.mfcc(y = signal, sr = sr,n_mfcc= config.N_MFCC , n_fft = config.N_FFT, hop_length =config.HOP_LENGTH)
    signal = signal.T
    signal = librosa.feature.delta(signal, order=2)
    normalized_signal = config.SCALER.fit_transform(signal)
    normalized_signal = torch.tensor(normalized_signal).unsqueeze(0).unsqueeze(0)
    mu, var, ind1, x1,  ind2, x2, ind3, x3 = vae.encoder(normalized_signal)
    z  = vae.bottleneck(mu, var, vae.normal_distribution)
    decoder_output  = vae.decoder(z, ind1, x1,  ind2, x2,  ind3, x3).squeeze()
    decoder_output = decoder_output.detach().cpu().numpy()
    signal  = tosignal(decoder_output)
    #orgsignal= tosignal(orgsignal)
    save_signals([signal],"predicted_sound", config.SAVE_PRED_SOUND)
    #save_signals([orgsignal], "original_sound", config.SAVE_PRED_SOUND_ORIGINAL)
test_results()