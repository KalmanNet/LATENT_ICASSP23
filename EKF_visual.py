## EKF_visual ##
import torch
import torch.nn as nn
#from filing_paths import path_model
from config import get_h_derivative, eps, H,F, encoded_dimention, y_size
import sys
import numpy as np
import time
import matplotlib.pyplot as plt
#sys.path.insert(1, path_model)
from main_AE import test_epoch, create_dataset_loader
import math

if torch.cuda.is_available():
    dev = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    dev = torch.device("cpu")
    #print("Running on the CPU")

class ExtendedKalmanFilter:

    def __init__(self, SystemModel,Lorenz_data_flag ,EKF_with_encoder_flag, model_encoder_trained, mode='full'):
        ########## training encoder with EKF ###########
        self.encoder = model_encoder_trained.double()
        self.EKF_with_encoder_flag=EKF_with_encoder_flag
        self.Lorenz_data_flag = Lorenz_data_flag
        #self.loss_EKF_with_encoser_for_training =  nn.MSELoss(reduction='mean')
        #self.optimizer=torch.optim.Adam(self.encoder.parameters(), lr=1e-3, weight_decay=1e-2)
        ##################################################
        #self.f = SystemModel.f
        self.F_func = SystemModel.F
        self.F = SystemModel.F
        self.m = SystemModel.m

        # Has to be transformed because of EKF non-linearity
        self.givenQ = SystemModel.givenQ

        self.h = SystemModel.h
        self.n = SystemModel.n
        self.H = SystemModel.H

        # Has to be transofrmed because of EKF non-linearity
        #self.R = SystemModel.realR
        self.givenR = SystemModel.givenR

        self.T = SystemModel.T
        self.T_test = SystemModel.T_test

        # Pre allocate KG array
        if EKF_with_encoder_flag:
            self.KG_array = torch.zeros((self.T_test, self.m, encoded_dimention))
        else:
            self.KG_array = torch.zeros((self.T_test, self.m, self.n))

        # Full knowledge about the model or partial? (Should be made more elegant)
        if (mode == 'full'):
            self.fString = 'ModAcc'
            self.hString = 'ObsAcc'
        elif (mode == 'partial'):
            self.fString = 'ModInacc'
            self.hString = 'ObsInacc'

    # Predict
    def Predict(self):
        ####### Predict the 1-st moment of x [F*x]
        self.x_t_given_prev = torch.matmul(self.F_func(self.x_t_est) ,self.x_t_est.double())
        #torch.squeeze(self.f(self.x_t_est.float()))
        #.type(torch.FloatTensor)
        # Predict the 1-st moment of y  [H*F*x]
        # Compute the Jacobians
        if self.EKF_with_encoder_flag: #H=I
            self.y_t_given_prev =  torch.matmul(self.H.double(), self.x_t_given_prev.double())
            self.UpdateJacobians(self.F_func(self.x_t_given_prev), self.H)
        else: #using real h function and its derivative for update stage
            self.y_t_given_prev = torch.squeeze(self.h(self.x_t_given_prev))
            self.UpdateJacobians(self.F_func(self.x_t_est),get_h_derivative(self.x_t_given_prev).view(-1, self.m))

        ####### Predict the 2-nd moment of x  cov(x)=[F*Cov(x)*F_T+Q]
        self.m2x_prior = torch.matmul(self.F, self.m2x_posterior.double())
        self.m2x_prior = torch.matmul(self.m2x_prior, self.F_T) + self.givenQ
        self.m2y = torch.matmul(self.H, self.m2x_prior)
        self.m2y = torch.matmul(self.m2y, self.H_T) + self.givenR

        # plt.imshow(self.y_t_given_prev)
        # plt.show()
        # plt.imshow(tmp.reshape(28,28))
        # plt.show()
        # plt.imshow(self.H.reshape(28,28))
        # plt.show()
        # Predict the 2-nd moment of y  cov(x)=[H*Cov(x)*H_T+R]

    # Compute the Kalman Gain
    def KGain(self):
        self.KG = torch.matmul(self.m2x_prior, self.H_T)
        if self.EKF_with_encoder_flag:
            self.KG = torch.matmul(self.KG.double().cpu(), torch.inverse(self.m2y.cpu().detach()+ eps * np.eye(encoded_dimention)))
        else:
            self.KG = torch.matmul(self.KG.double().cpu(), torch.inverse(self.m2y.cpu()+eps*np.eye(self.n)))
        # Save KalmanGain
        self.KG_array[self.i] = self.KG
        self.i += 1

    # Innovation
    def Innovation(self, y):
        if self.EKF_with_encoder_flag:
            self.dy = y - self.y_t_given_prev
        else:
            self.dy = y - self.y_t_given_prev.reshape(y_size*y_size) #after h function need to be flatten

    # Compute Posterior
    def Correct(self):
        # Compute the 1-st posterior moment
        self.x_t_est = self.x_t_given_prev + torch.matmul(self.KG.to(dev),self.dy.double().to(dev))
        # Compute the 2-nd posterior moment
        self.m2x_posterior = torch.matmul(self.m2y.double().to(dev), torch.transpose(self.KG, 0, 1).to(dev))
        self.m2x_posterior = self.m2x_prior - torch.matmul(self.KG.to(dev), self.m2x_posterior.to(dev))

    def Update(self, y):
        self.Predict()
        #t_start = time.time()
        self.KGain()
        #print('calculate KGain took {}'.format(time.time()-t_start))
        self.Innovation(y)
        self.Correct()
        return self.x_t_est, self.m2x_posterior

    def InitSequence_EKF(self, m1x_0, m2x_0):
        self.m1x_0 = m1x_0
        self.m2x_0 = m2x_0

        #########################

    def UpdateJacobians(self, F, H):
        self.F = F.double()
        self.F_T = torch.transpose(F, 0, 1).double()
        self.H = H.double()
        self.H_T = torch.transpose(H, 0, 1).double()
        # print(self.H,self.F,'\n')

    ### Generate Sequence ###
    #########################
    def GenerateSequence(self, video, target_of_video, T):
        loss_fn = nn.MSELoss(reduction='mean')
        # test_loader = create_dataset_loader(video.cpu().detach().numpy(), target_of_video.cpu().detach().numpy(), True,32)
        # test_loss = test_epoch(self.encoder, self.encoder, test_loader, torch.nn.MSELoss(), 32, True, not True)
        # print(10 * math.log10(test_loss))

        # Pre allocate an array for predicted state and variance
        self.x = torch.empty(size=[self.m, T]) #space for estimation of state sequence
        self.encoder_output = torch.empty(size=[self.m, T])
        self.sigma = torch.empty(size=[self.m, self.m, T]) #covariance state noise
        # Pre allocate KG array
        if self.EKF_with_encoder_flag:
            self.KG_array = torch.zeros((T, self.m, encoded_dimention))   # space for KG of each time step
        else:
            self.KG_array = torch.zeros((T, self.m, self.n))
        self.i = 0  # Index for KG_array alocation
        self.x_t_est = torch.squeeze(self.m1x_0)
        self.m2x_posterior = torch.squeeze(self.m2x_0)

        for t in range(0, T):
            obsevation_t = video[:, t]  # taking the t'th image
            if self.EKF_with_encoder_flag:  # working on images so need to go throgh an encoder
                # imgplot = plt.imshow(yt.reshape(y_size, y_size).cpu().detach().numpy())
                # plt.title("obsevation")
                # plt.colorbar()
                # plt.show()
                obsevation_t = obsevation_t.reshape(1, 1, y_size, y_size)
                self.encoder.eval()
                encoder_output_t=torch.squeeze(self.encoder(obsevation_t.double()))
                self.encoder_output[:, t] = encoder_output_t
                #target_t = target_of_video[:,t]
                xt, sigmat = self.Update(encoder_output_t)
            else:
                xt, sigmat = self.Update(obsevation_t)
            self.sigma[:, :, t] = torch.squeeze(sigmat)
            self.x[:, t] = torch.squeeze(xt)
