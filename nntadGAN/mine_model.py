#!/usr/bin/env python
# coding: utf-8

import os
import logging
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from dtaidistance import dtw

class Encoder(nn.Module):

    def __init__(self, encoder_path, signal_shape=100):
        super(Encoder, self).__init__()
        self.signal_shape = signal_shape
        self.lstm = nn.LSTM(input_size=self.signal_shape, hidden_size=20, num_layers=1, bidirectional=True)
        self.dense = nn.Linear(in_features=40, out_features=20)
        self.encoder_path = encoder_path

    def forward(self, x):
        x = x.view(1, 64, self.signal_shape).float()
        x, (hn, cn) = self.lstm(x)
        x = self.dense(x)
        return (x)

class Decoder(nn.Module):
    def __init__(self, decoder_path, signal_shape=100):
        super(Decoder, self).__init__()
        self.signal_shape = signal_shape
        self.lstm = nn.LSTM(input_size=20, hidden_size=64, num_layers=2, bidirectional=True)
        self.dense = nn.Linear(in_features=128, out_features=self.signal_shape)
        self.decoder_path = decoder_path

    def forward(self, x):
        x, (hn, cn) = self.lstm(x)
        x = self.dense(x)
        return (x)

class CriticX(nn.Module):
    def __init__(self, critic_x_path, signal_shape=100):
        super(CriticX, self).__init__()
        self.signal_shape = signal_shape
        self.dense1 = nn.Linear(in_features=self.signal_shape, out_features=20)
        self.dense2 = nn.Linear(in_features=20, out_features=1)
        self.critic_x_path = critic_x_path

    def forward(self, x):
        x = x.view(1, 64, self.signal_shape).float()
        x = self.dense1(x)
        x = self.dense2(x)
        return (x)

class CriticZ(nn.Module):
    def __init__(self, critic_z_path):
        super(CriticZ, self).__init__()
        self.dense1 = nn.Linear(in_features=20, out_features=1)
        self.critic_z_path = critic_z_path

    def forward(self, x):
        x = self.dense1(x)
        return (x)

def unroll_signal(self, x):
    x = np.array(x).reshape(100)
    return np.median(x)

def dtw_reconstruction_error(x, x_hat):
    """
    Calculate the DTW reconstruction error between original and reconstructed signals.
    
    Parameters:
    x (np.array): Original signal
    x_hat (np.array): Reconstructed signal
    
    Returns:
    float: DTW reconstruction error
    """
    return dtw.distance(x, x_hat)

def test(encoder, decoder, critic_x, df):
    """
    Returns a dataframe with original value, reconstructed value, reconstruction error, critic score
    """
    X_ = list()
    RE = list()  # Reconstruction error
    CS = list()  # Critic score

    for i in range(0, df.shape[0]):
        x = df.rolled_signal[i]
        x = torch.tensor(x, dtype=torch.float32).view(1, 64, 100)
        z = encoder(x)
        x_ = decoder(z)

        re = dtw_reconstruction_error(x_.detach().numpy().flatten(), x.detach().numpy().flatten())  # reconstruction error
        cs = critic_x(x)
        cs = cs.detach().numpy().flatten()[0]
        RE.append(re)
        CS.append(cs)

        x_ = unroll_signal(x_.detach().numpy().flatten())

        X_.append(x_)

    df['generated_signals'] = X_
    df['reconstruction_error'] = RE
    df['critic_score'] = CS

    return df

# Carregar seu dataset
df = pd.read_csv('meu_arquivo_massflow_A1_csv.csv')
# Pre-processar seu dataset conforme necessário
# ...

# Ajuste os caminhos dos modelos
encoder_path = 'encoder1.pt'
decoder_path = 'decoder1.pt'
critic_x_path = 'critic_x1.pt'
critic_z_path = 'critic_z1.pt'

# Instanciar os modelos com os caminhos
#encoder = Encoder(encoder_path)
#decoder = Decoder(decoder_path)
#critic_x = CriticX(critic_x_path)
#critic_z = CriticZ(critic_z_path)

# Carregar pesos pré-treinados
# encoder.load_state_dict(torch.load(encoder_path))
# decoder.load_state_dict(torch.load(decoder_path))
# critic_x.load_state_dict(torch.load(critic_x_path))
# critic_z.load_state_dict(torch.load(critic_z_path))

# Fazer o teste
# resultado = test(encoder, decoder, critic_x, df)
# print(resultado.head())