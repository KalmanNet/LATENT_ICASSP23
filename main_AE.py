###### main_AE ################
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from config import *
#from visual_supplementary import create_dataset, visualize_similarity
from Extended_data_visual import DataLoader_GPU
import numpy as np
import math
import random

def train(path_enc,encoder, train_loader, val_loader, num_epochs, batch_size, flag_prior):
    encoder.train()
    encoder.float()
    best_val_loss=1000000
    ############ Learning Configurations ###############
    loss_fn = torch.nn.MSELoss()
    params_to_optimize = [{'params': encoder.parameters()}]
    optimizer = torch.optim.Adam(params_to_optimize, lr=lr, weight_decay=wd)
    train_loss_epoch=[]
    val_loss_epoch = []
    for epoch in range(num_epochs):
        train_loss = []
        for k, batch in enumerate(train_loader):
            if (batch[0].shape[0] == batch_size):  # taking only full batch to learn from
                image_batch = batch[0]
                states_with_noise_batch = batch[1]
                targets_batch = batch[2]
                if flag_prior:
                    encoded_data = encoder(image_batch,states_with_noise_batch)
                else:
                    encoded_data = encoder(image_batch)
                loss = loss_fn(encoded_data.float(), targets_batch.float())
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # Print batch loss
                train_loss.append(loss.detach().cpu().numpy())
        train_loss_epoch.append(10 * math.log10(np.mean(train_loss)))
        epoch_val_loss = test_epoch(encoder, val_loader, loss_fn, batch_size, flag_prior)
        val_loss_epoch.append(10 * math.log10(epoch_val_loss))
        print('epoch: {} train loss: {} dB val loss: {} dB'.format(epoch, train_loss_epoch[epoch],val_loss_epoch[epoch]))
        if val_loss_epoch[epoch] < best_val_loss:
            torch.save(encoder.state_dict(), path_enc)

    return encoder, val_loss_epoch, train_loss_epoch

def get_encoder_output(test_input, model_encoder_trained):
    x_out_test_all = torch.empty(test_input.shape[0], m, test_input.shape[2])
    for j in range(0, test_input.shape[0]):
        objervation_sample = test_input[j, :, :]
        encoder_sample_output = torch.empty(m, test_input.shape[2])
        for t in range(0, test_input.shape[2]):
            AE_input = objervation_sample[:, t].reshape(1, 1, y_size, y_size)
            encoder_sample_output[:,t] = model_encoder_trained(AE_input).squeeze()
        x_out_test_all[j,:,:]=encoder_sample_output
    return x_out_test_all

def test_epoch(encoder, val_loader, loss_fn, batch_size, flag_prior):
    #encoder_test = torch.empty(100, 3, 2000)
    encoder.eval()
    with torch.no_grad(): # No need to track the gradients
        val_loss = []
        for k, batch in enumerate(val_loader):
            if (batch[0].shape[0] == batch_size):  # taking only full batch to learn from
                image_batch = batch[0]
                states_with_noise_batch = batch[1]
                targets_batch = batch[2]
                if flag_prior:
                    encoded_data = encoder(image_batch,states_with_noise_batch)
                else:
                    encoded_data = encoder(image_batch)
                loss = loss_fn(encoded_data.float(), targets_batch.float())
                if torch.isnan(loss):
                    print("we have nan")
                val_loss.append(loss.detach().cpu().numpy())
    return np.mean(val_loss)

def inference_with_prior(encoder, test_input, test_target):
    encoder.eval()
    loss_fn = torch.nn.MSELoss()
    with torch.no_grad(): # No need to track the gradients
        test_loss = []
        for k,trajectory_obs in enumerate(test_input):
            trajectory_target = test_target[k]
            state_prev = torch.ones(test_target.shape[1], 1)
            x_out = torch.empty(test_target.shape[1], trajectory_obs.shape[1])
            for t in range(trajectory_obs.shape[1]):
                obs = trajectory_obs[:,t].reshape(1, 1, y_size, y_size)
                prior = torch.matmul(F(state_prev).float(), state_prev)
                encoded_data = encoder(obs, prior.transpose(0,1))
                x_out[:, t] = encoded_data
                state_prev = encoded_data.transpose(0, 1)
            test_loss.append(loss_fn(x_out, trajectory_target).detach().cpu().numpy())
        return np.mean(test_loss)

class Encoder_synt(nn.Module):
    def __init__(self,encoded_space_dim):
        super(Encoder_synt, self).__init__()
        self.encoder = nn.Sequential(
            torch.nn.Linear(y_size * y_size, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(64, encoded_space_dim))
    def forward(self, x):
        x = self.encoder(x)
        return x

class Decoder_synt(nn.Module):
    def __init__(self, encoded_space_dim):
        super(Decoder_synt, self).__init__()
        self.decoder = nn.Sequential(
            torch.nn.Linear(encoded_space_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, y_size * y_size),
            torch.nn.Sigmoid())

    def forward(self, x):
        x = self.decoder(x)
        return x

class Encoder_conv(nn.Module):
    def __init__(self, encoded_space_dim):
        super(Encoder_conv, self).__init__()
        ### Convolutional section
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(True))
        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        ### Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(288, 32),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(32, encoded_space_dim))

    def forward(self, x):
        #x=x.type(torch.DoubleTensor)
        x = self.encoder_conv(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x


class Encoder_conv_with_prior(nn.Module):
    def __init__(self, encoded_space_dim):
        super(Encoder_conv_with_prior, self).__init__()
        ### Convolutional section
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(True))
        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        ### Linear section
        self.prior_fc = nn.Sequential(
            nn.Linear(3, 8),
            #nn.ReLU(True),
            nn.Dropout(0.5))
        self.encoder_lin = nn.Sequential(
            nn.Linear(296, 32),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(32, encoded_space_dim))

    def forward(self, x, prior):
        #x=x.type(torch.DoubleTensor)
        x = self.encoder_conv(x)
        x = self.flatten(x)
        prior = self.prior_fc(prior)
        comb = torch.cat((prior, x), 1)
        out = self.encoder_lin(comb)
        return out

class Decoder_conv(nn.Module):
    def __init__(self, encoded_space_dim):
        super(Decoder_conv, self).__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 32),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(32, 128),
            nn.ReLU(True))
        self.unflatten = nn.Unflatten(dim=1,unflattened_size=(32, 2, 2))
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=0, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.ConvTranspose2d(8, 1, 3, stride=2,padding=1, output_padding=1))

    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        #x = torch.sigmoid(x)
        return x

def check_learning_process(img_batch,recon_batch,epoch, name):
    y_nump = img_batch[32].reshape(28,28).detach().numpy()
    #reshape(24,24).detach().numpy().squeeze()
    y_recon_nump = recon_batch[32].reshape(28,28).detach().numpy()
    #recon_batch[32].reshape(24,24).detach().numpy().squeeze()
    fig = plt.figure(figsize=(10, 7))
    fig.add_subplot(1, 2, 1)
    plt.imshow(y_nump)
    #cmap = 'gray'
    plt.axis('off')
    plt.title("origin")

    fig.add_subplot(1, 2, 2)
    plt.imshow(y_recon_nump)
    #, cmap='gray'
    plt.axis('off')
    plt.title("reconstruct")
    fig.savefig('AE Process/{} Process at epoch {}.PNG'.format(name, epoch))

def print_process(val_loss_list, train_loss_list , flag_only_encoder):
    if flag_only_encoder:
        configuration='only encoder'
    else:
        configuration='Auto Encoder'
    fig = plt.figure(figsize=(10, 7))
    fig.add_subplot(1, 1, 1)
    plt.plot(train_loss_list, label='train')
    plt.plot(val_loss_list, label='val')
    plt.title("Loss of "+configuration)
    plt.xlabel("epochs")
    plt.ylabel("MSE [dB]")
    plt.legend()
    plt.savefig(r".\AE Process\Learning_process encoder with prior r={} prior r={}.jpg".format(real_r,prior_r))

def create_dataset_loader(imgs_np, states_np_with_noise, states_np, batch_size):
    # create imgs list
    data_list = []
    for k in range(imgs_np.shape[0]):
        sample = imgs_np[k]#test set
        seq_length = sample.shape[1]
        for t in range(seq_length):
            img = sample[:, t].reshape((1, 28, 28))
            data_list.append(img)

    # create states with noise list
    data_state_noise_list = []
    for k in range(states_np_with_noise.shape[0]):
        sample = states_np_with_noise[k]#test set
        for t in range(seq_length):
            state_with_noise = sample[:, t]
            data_state_noise_list.append(state_with_noise)

    # create target list ##
    chosen_targets_np= states_np
    targets_list = []
    for k in range(chosen_targets_np.shape[0]):
        sample = chosen_targets_np[k]
        for t in range(seq_length):
            target = sample[:,t]
            targets_list.append(target)

    dataset=[]
    for k in range(len(data_list)):
        dataset.append((data_list[k],data_state_noise_list[k],targets_list[k]))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    return loader

def add_noise(states_np_train,r):
    stats_np_with_noise = np.empty_like(states_np_train)
    for k in range(states_np_train.shape[0]): #runing over each trajectory
        for i in range(states_np_train.shape[2]):#running over each time step
            added_noise = np.random.normal(0, r, states_np_train.shape[1])
            stats_np_with_noise[k,:,i] = states_np_train[k,:,i]+added_noise
    return stats_np_with_noise

def initialize_data_AE(path_for_data, batch_size,prior_r):
    ### load data ###
    [train_input, train_target, cv_input, cv_target, test_input, test_target] = DataLoader_GPU(path_for_data)
    ############ create datasets ####################
    imgs_np_train=train_input.cpu().detach().numpy()
    states_np_train=train_target.cpu().detach().numpy()
    stats_np_with_noise_train=add_noise(states_np_train,prior_r)
    train_loader = create_dataset_loader(imgs_np_train, stats_np_with_noise_train, states_np_train, batch_size)

    imgs_np_val=cv_input.cpu().detach().numpy()
    states_np_val=cv_target.cpu().detach().numpy()
    stats_np_with_noise_val = add_noise(states_np_val, prior_r)
    val_loader = create_dataset_loader(imgs_np_val, stats_np_with_noise_val, states_np_val, batch_size)

    imgs_np_test=test_input.cpu().detach().numpy()
    states_np_test=test_target.cpu().detach().numpy()
    stats_np_with_noise_test = add_noise(states_np_test, prior_r)
    test_loader = create_dataset_loader(imgs_np_test, stats_np_with_noise_test, states_np_test, batch_size)
    return train_loader,val_loader,test_loader,test_input, test_target

## Hypper parameters
lr = 1e-4
wd = 0.01
d = 3
batch_size = 512
num_epochs = 300

if __name__ == '__main__':
    flag_only_encoder = True
    flag_data_syntetic = False
    torch.manual_seed(0)
    random.seed(0)

    flag_prior = True
    flag_inference_prior = True
    flag_decimated = False
    flag_train = False
    for prior_r in [4,2]:
        print("prior r is {}".format(prior_r))
        for r in [1,2,4,10]:
            ############ data sinerio #################
            if flag_decimated:
                path_for_data = r'.\Simulations\Lorenz_visual\lorenz_T=200_decimated_q=0.1_r={}.pt'.format(r)
            else:
                path_for_data = r'.\Simulations\Lorenz_visual\lorenz_T=200_q=0.1_r={}.pt'.format(r)
            ### load data and define setup
            train_loader,valid_loader, test_loader, test_input, test_target= initialize_data_AE(path_for_data, batch_size,prior_r)

            ### choosing model
            if flag_prior:
                encoder = Encoder_conv_with_prior(encoded_space_dim=d)
                if flag_decimated:
                    path_enc = r'.\saved_models\lorenz_Only_encoder_decimated_r={}_prior_r={}.pt'.format(r, prior_r)
                else:
                    path_enc = r'.\saved_models\lorenz_Only_encoder_r={}_prior_r={}.pt'.format(r,prior_r)
            else:
                encoder = Encoder_conv(encoded_space_dim=d)
                if flag_decimated:
                    path_enc = r".\saved_models\lorenz_Only_encoder_decimated_r={}.pt".format(r)
                else:
                    path_enc = r".\saved_models\lorenz_Only_encoder_r={}.pt".format(r)
            print("Model Parameters lr = {} wd = {} ".format(lr, wd))

            ### Train
            if flag_train:
                encoder, val_loss_epoch, train_loss_epoch = train(path_enc,encoder, train_loader,valid_loader, num_epochs, batch_size, flag_prior)
                print_process(val_loss_epoch, train_loss_epoch, flag_only_encoder)
            else:
                encoder.load_state_dict(torch.load(path_enc), strict=False)

            ### test model
            if flag_inference_prior:
                test_loss = inference_with_prior(encoder, test_input, test_target)
            else:
                test_loss = test_epoch(encoder, test_loader, torch.nn.MSELoss(), batch_size, flag_prior)
            print("Test loss of prior r {} and obs r {} is: {} dB".format(prior_r,r,10 * math.log10(test_loss)))



