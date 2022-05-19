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


plot()