#!/usr/bin/env python
# coding: utf-8

import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import mine_model
import mine_anomaly_detection

from sklearn.model_selection import train_test_split
from tqdm import tqdm

if torch.cuda.is_available():
    device = torch.device("cuda")  # Usa a GPU
    print(f"Usando GPU: {torch.cuda.get_device_name(device)}")
else:
    device = torch.device("cpu")  # Usa a CPU
    print("Usando CPU")

logging.basicConfig(filename='train.log', level=logging.DEBUG)

class SignalDataset(Dataset):
    def __init__(self, path):
        self.signal_df = pd.read_csv(path)
        self.signal_columns = self.make_signal_list()
        self.make_rolling_signals()

    def make_signal_list(self):
        signal_list = list()
        for i in range(-50, 50):
            signal_list.append('signal'+str(i))
        return signal_list

    def make_rolling_signals(self):
        for i in range(-50, 50):
            self.signal_df['signal'+str(i)] = self.signal_df['signal'].shift(i)
        self.signal_df = self.signal_df.dropna()
        self.signal_df = self.signal_df.reset_index(drop=True)

    def __len__(self):
        return len(self.signal_df)

    def __getitem__(self, idx):
        row = self.signal_df.loc[idx]
        x = row[self.signal_columns].values.astype(float)
        x = torch.from_numpy(x)
        return {'signal':x, 'anomaly':row['anomaly']}

def critic_x_iteration(sample):
    optim_cx.zero_grad()

    x = sample['signal'].view(1, batch_size, signal_shape)
    valid_x = critic_x(x)
    valid_x = torch.squeeze(valid_x)
    critic_score_valid_x = torch.mean(torch.ones(valid_x.shape) * valid_x) #Wasserstein Loss

    #The sampled z are the anomalous points - points deviating from actual distribution of z (obtained through encoding x)
    z = torch.empty(1, batch_size, latent_space_dim).uniform_(0, 1)
    x_ = decoder(z)
    fake_x = critic_x(x_)
    fake_x = torch.squeeze(fake_x)
    critic_score_fake_x = torch.mean(torch.ones(fake_x.shape) * fake_x)  #Wasserstein Loss

    alpha = torch.rand(x.shape)
    ix = Variable(alpha * x + (1 - alpha) * x_) #Random Weighted Average
    ix.requires_grad_(True)
    v_ix = critic_x(ix)
    v_ix.mean().backward()
    gradients = ix.grad
    #Gradient Penalty Loss
    gp_loss = torch.sqrt(torch.sum(torch.square(gradients).view(-1)))

    #Critic has to maximize Cx(Valid X) - Cx(Fake X).
    #Maximizing the above is same as minimizing the negative.
    wl = critic_score_fake_x - critic_score_valid_x
    loss = wl + gp_loss
    loss.backward()
    optim_cx.step()

    return loss

def critic_z_iteration(sample):
    optim_cz.zero_grad()

    x = sample['signal'].view(1, batch_size, signal_shape)
    z = encoder(x)
    valid_z = critic_z(z)
    valid_z = torch.squeeze(valid_z)
    critic_score_valid_z = torch.mean(torch.ones(valid_z.shape) * valid_z)

    z_ = torch.empty(1, batch_size, latent_space_dim).uniform_(0, 1)
    fake_z = critic_z(z_)
    fake_z = torch.squeeze(fake_z)
    critic_score_fake_z = torch.mean(torch.ones(fake_z.shape) * fake_z) #Wasserstein Loss

    wl = critic_score_fake_z - critic_score_valid_z

    alpha = torch.rand(z.shape)
    iz = Variable(alpha * z + (1 - alpha) * z_) #Random Weighted Average
    iz.requires_grad_(True)
    v_iz = critic_z(iz)
    v_iz.mean().backward()
    gradients = iz.grad
    gp_loss = torch.sqrt(torch.sum(torch.square(gradients).view(-1)))

    loss = wl + gp_loss
    loss.backward()
    optim_cz.step()

    return loss

def encoder_iteration(sample):
    optim_enc.zero_grad()

    x = sample['signal'].view(1, batch_size, signal_shape)
    valid_x = critic_x(x)
    valid_x = torch.squeeze(valid_x)
    critic_score_valid_x = torch.mean(torch.ones(valid_x.shape) * valid_x) #Wasserstein Loss

    z = torch.empty(1, batch_size, latent_space_dim).uniform_(0, 1)
    x_ = decoder(z)
    fake_x = critic_x(x_)
    fake_x = torch.squeeze(fake_x)
    critic_score_fake_x = torch.mean(torch.ones(fake_x.shape) * fake_x)

    enc_z = encoder(x)
    gen_x = decoder(enc_z)

    mse = mse_loss(x.float(), gen_x.float())
    loss_enc = mse + critic_score_valid_x - critic_score_fake_x
    loss_enc.backward(retain_graph=True)
    optim_enc.step()

    return loss_enc

def decoder_iteration(sample):
    optim_dec.zero_grad()

    x = sample['signal'].view(1, batch_size, signal_shape)
    z = encoder(x)
    valid_z = critic_z(z)
    valid_z = torch.squeeze(valid_z)
    critic_score_valid_z = torch.mean(torch.ones(valid_z.shape) * valid_z)

    z_ = torch.empty(1, batch_size, latent_space_dim).uniform_(0, 1)
    fake_z = critic_z(z_)
    fake_z = torch.squeeze(fake_z)
    critic_score_fake_z = torch.mean(torch.ones(fake_z.shape) * fake_z)

    enc_z = encoder(x)
    gen_x = decoder(enc_z)

    mse = mse_loss(x.float(), gen_x.float())
    loss_dec = mse + critic_score_valid_z - critic_score_fake_z
    loss_dec.backward(retain_graph=True)
    optim_dec.step()

    return loss_dec


def train(n_epochs=2000):
    logging.debug('Starting training')
    cx_epoch_loss = list()
    cz_epoch_loss = list()
    encoder_epoch_loss = list()
    decoder_epoch_loss = list()

    for epoch in tqdm(range(n_epochs),total=n_epochs):
        logging.debug('Epoch {}'.format(epoch))
        n_critics = 5

        cx_nc_loss = list()
        cz_nc_loss = list()

        for i in range(n_critics):
            cx_loss = list()
            cz_loss = list()

            for batch, sample in enumerate(train_loader):
                loss = critic_x_iteration(sample)
                cx_loss.append(loss)

                loss = critic_z_iteration(sample)
                cz_loss.append(loss)

            cx_nc_loss.append(torch.mean(torch.tensor(cx_loss)))
            cz_nc_loss.append(torch.mean(torch.tensor(cz_loss)))

        logging.debug('Critic training done in epoch {}'.format(epoch))
        
        encoder_loss = list()
        decoder_loss = list()

        for batch, sample in enumerate(train_loader):
            enc_loss = encoder_iteration(sample)
            dec_loss = decoder_iteration(sample)
            encoder_loss.append(enc_loss)
            decoder_loss.append(dec_loss)

        cx_epoch_loss.append(torch.mean(torch.tensor(cx_nc_loss)))
        cz_epoch_loss.append(torch.mean(torch.tensor(cz_nc_loss)))
        encoder_epoch_loss.append(torch.mean(torch.tensor(encoder_loss)))
        decoder_epoch_loss.append(torch.mean(torch.tensor(decoder_loss)))
        logging.debug('Encoder decoder training done in epoch {}'.format(epoch))
        logging.debug('critic x loss {:.3f} critic z loss {:.3f} \nencoder loss {:.3f} decoder loss {:.3f}\n'.format(cx_epoch_loss[-1], cz_epoch_loss[-1], encoder_epoch_loss[-1], decoder_epoch_loss[-1]))

        if epoch % 10 == 0:
            torch.save(encoder.state_dict(), encoder.encoder_path)
            torch.save(decoder.state_dict(), decoder.decoder_path)
            torch.save(critic_x.state_dict(), critic_x.critic_x_path)
            torch.save(critic_z.state_dict(), critic_z.critic_z_path)

    # Plotting the loss for each epoch
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(cx_epoch_loss)), cx_epoch_loss, label='Critic X Loss')
    plt.plot(range(len(cz_epoch_loss)), cz_epoch_loss, label='Critic Z Loss')
    plt.plot(range(len(encoder_epoch_loss)), encoder_epoch_loss, label='Encoder Loss')
    plt.plot(range(len(decoder_epoch_loss)), decoder_epoch_loss, label='Decoder Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss for each Epoch')
    plt.legend()
    plt.show()
    
if __name__ == "__main__":

    TRAIN = True

    #dataset = pd.read_csv(r"C:\Users\pedro\OneDrive\Documents\GitHub\RunningIn_DatabaseFunc\nntadGAN\exchange-2_cpc_results.csv")
    #dataset = pd.read_csv(r'nntadGAN\meu_arquivo_massflow_A1_csv.csv')

    dataset = pd.read_csv(r'nntadGAN\anomalias_visiveis.csv')
    

    #Splitting intro train and test
    #TODO could be done in a more pythonic way
    train_len = int(0.7 * dataset.shape[0])

    (train_dataset,test_dataset) = train_test_split(dataset, test_size=0.3, stratify=dataset["anomaly"])
    train_dataset.to_csv('train_dataset.csv', index=False)
    test_dataset.to_csv('test_dataset.csv', index=False)

    encoder_path = 'encoder1.pt'
    decoder_path = 'decoder1.pt'
    critic_x_path = 'critic_x1.pt'
    critic_z_path = 'critic_z1.pt'

    lr = 1e-6
    signal_shape = 100
    latent_space_dim = 20
    batch_size = 64

    if TRAIN:

        train_dataset = SignalDataset(path='train_dataset.csv')
        test_dataset = SignalDataset(path='test_dataset.csv')
        train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True)

        logging.info('Number of train datapoints is {}'.format(len(train_dataset)))
        logging.info('Number of samples in train dataset {}'.format(len(train_dataset)))
    
        encoder = mine_model.Encoder(encoder_path, signal_shape)
        decoder = mine_model.Decoder(decoder_path, signal_shape)
        critic_x =mine_model.CriticX(critic_x_path, signal_shape)
        critic_z = mine_model.CriticZ(critic_z_path)

        mse_loss = torch.nn.MSELoss()

        optim_enc = optim.Adam(encoder.parameters(), lr=lr, betas=(0.5, 0.999))
        optim_dec = optim.Adam(decoder.parameters(), lr=lr, betas=(0.5, 0.999))
        optim_cx = optim.Adam(critic_x.parameters(), lr=lr, betas=(0.5, 0.999))
        optim_cz = optim.Adam(critic_z.parameters(), lr=lr, betas=(0.5, 0.999))

        train(n_epochs=5)

    else:
        encoder = mine_model.Encoder(encoder_path, signal_shape)
        decoder = mine_model.Decoder(decoder_path, signal_shape)
        critic_x =mine_model.CriticX(critic_x_path, signal_shape)
        critic_z = mine_model.CriticZ(critic_z_path)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True)

    mine_anomaly_detection.test(test_loader, encoder, decoder, critic_x)