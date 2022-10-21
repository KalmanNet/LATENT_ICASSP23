## Pipeline_KF_visual ##
import torch
import torch.nn as nn
import random
#from Plot import Plot
import time
import matplotlib.pyplot as plt
from config import y_size, F
class Pipeline_KF:
    def __init__(self, Time, sinerio,folderName_KNet, data_name, fix_H_flag, fix_encoder_flag, lorenz_data_flag,real_r):
        super().__init__()
        self.Time = Time                                                     # Time date and clock
        self.folderName = folderName_KNet + '/'                              # kalman pipeline folder
        #self.Learning_process_folderName = folderName_Learning_process + '/' # Learning Process folder
        ### model names - KGain and all Pipeline ###
        self.modelName = data_name
        self.modelFileName = self.folderName + "KGain_optimal_" + self.modelName + "_"+sinerio+"_fix_enc_{}_fix_h_{}_r_{}.pt".format(0,0,real_r)
        #self.PipelineName = self.folderName + "pipeline_" + + self.modelName + "_fix_enc_{}_r_{}.pt".format(fix_encoder_flag,real_r)
        #### flags for the process ######
        self.fix_H_flag= fix_H_flag                  # is H matrix in latent space fix?
        self.lorenz_data_flag = lorenz_data_flag # is it pendulum data?
        self.fix_encoder_flag = fix_encoder_flag     # is the encoder fix?

    def save(self):
        Itay=1
        #torch.save(self, self.PipelineName)

    def setssModel(self, ssModel):
        self.ssModel = ssModel

    def setModel(self, KNet_model):
        self.model = KNet_model

    def setTrainingParams(self, n_Epochs, n_Batch, learningRate, weightDecay):
        self.N_Epochs = n_Epochs  # Number of Training Epochs
        self.N_B = n_Batch # Number of Samples in Batch
        self.learningRate = learningRate # Learning Rate
        self.weightDecay = weightDecay # L2 Weight Regularization - Weight Decay

        # MSE LOSS Function
        self.loss_fn = nn.MSELoss(reduction='mean')

        # Use the optim package to define an Optimizer that will update the weights of
        # the model for us. Here we will use Adam; the optim package contains many other
        # optimization algoriths. The first argument to the Adam constructor tells the
        # optimizer which Tensors it should update.
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learningRate, weight_decay=self.weightDecay)

    def check_tracking(self, ti, knet, target,  EKF, encoder_output, folder_learning_path):
        ## Check Changes
        if self.lorenz_data_flag:
            fig, (ax1, ax2, ax3) = plt.subplots(3)
        else:
            fig, (ax1, ax2) = plt.subplots(2)

        fig.suptitle("components Estimation in epoch {}".format(ti))

        KNet_est_0 = knet[0, :].cpu().detach().numpy()
        GT_0 = target[0, :]
        Enc_0 = encoder_output[0, :].cpu().detach().numpy()
        EKF_0 = EKF[0, :].cpu().detach().numpy()
        axis = list(range(GT_0.shape[0]))
        ax1.plot(axis, GT_0)
        ax1.plot(axis, KNet_est_0)
        ax1.plot(axis, Enc_0)
        ax1.plot(axis, EKF_0)
        ax1.legend(["GT", "KNet", "Encoder", "EKF"])

        KNet_est_1 = knet[1, :].cpu().detach().numpy()
        GT_1 = target[1, :]
        Enc_1 = encoder_output[1, :].cpu().detach().numpy()
        EKF_1 = EKF[1, :].cpu().detach().numpy()
        axis = list(range(GT_1.shape[0]))
        ax2.plot(axis, GT_1)
        ax2.plot(axis, KNet_est_1)
        ax2.plot(axis, Enc_1)
        ax2.plot(axis, EKF_1)
        ax2.legend(["GT", "KNet", "Encoder", "EKF"])

        if self.lorenz_data_flag:
            KNet_est_2 = knet[2, :].cpu().detach().numpy()
            GT_2 = target[2, :]
            Enc_2 = encoder_output[2, :].cpu().detach().numpy()
            EKF_2 = EKF[2, :].cpu().detach().numpy()
            axis = list(range(GT_2.shape[0]))
            ax3.plot(axis, GT_2)
            ax3.plot(axis, KNet_est_2)
            ax3.plot(axis, Enc_2)
            ax3.plot(axis, EKF_2)
            ax3.legend(["GT", "KNet", "Encoder", "EKF"])

        fig.show()
        fig.savefig(folder_learning_path + "/1D epoch {}.png".format(ti))

        if self.lorenz_data_flag:
            fig, ax = plt.subplots(2,2,figsize=(10,10),subplot_kw=dict(projection='3d'))
            fig.suptitle("components 3D comparison in epoch {}".format(ti))
            KNet_3D = ax[0,0].plot(KNet_est_0, KNet_est_1, KNet_est_2)
            ax[0,0].set_title("KNet")
            Enc_3D = ax[0,1].plot(Enc_0, Enc_1, Enc_2)
            ax[0,1].set_title("Encoder")
            GT_3D = ax[1,0].plot(GT_0, GT_1, GT_2)
            ax[1,0].set_title("GT")
            EKF_3D = ax[1,1].plot(EKF_0, EKF_1, EKF_2)
            ax[1,1].set_title("EKF with Encoder")
            fig.show()
            fig.savefig(folder_learning_path + "/3D epoch {}.png".format(ti))

    def get_seq_knet_output(self,observation_seq , target, prior_flag):
        length_seq = observation_seq.shape[1]
        x_out = torch.empty(self.ssModel.m, length_seq)
        z_encoder_output = torch.empty(self.ssModel.m, length_seq)
        state_prev = torch.ones(self.ssModel.m,1) #[3,1]
        for t in range(0, length_seq):
            AE_input = observation_seq[:,t].reshape(1, 1, y_size, y_size)
            prior = torch.matmul(F(state_prev).float(), state_prev) #[3,1] IN AND OUT
            self.model.model_encoder = self.model.model_encoder.float()
            if prior_flag:
                y_decoaded = self.model.model_encoder(AE_input,prior.transpose(0,1)).squeeze() #[3,]
            else:
                y_decoaded = self.model.model_encoder(AE_input).squeeze()
            state_prev = y_decoaded.unsqueeze(1)
            z_encoder_output[:,t] = y_decoaded
            x_out[:,t]= self.model(y_decoaded)
            # if t%201==0:
            #     print('t= {} KNet output {} Encoder output {} Target {}'.format(t, x_out_training[:, t],z_encoder_output[:, t],target[:,t]))
        return x_out, z_encoder_output

    def NNTrain(self, n_Examples, train_input, train_target, n_CV, cv_input, cv_target, title, prior_flag):
        ####################### freezing weights ############################
        if  self.fix_encoder_flag:
            for param in self.model.parameters():
                param.requires_grad = True
            if self.fix_H_flag:
                for param in self.model.h_learned_lower_dimension.parameters():
                    param.requires_grad = False
            if self.fix_encoder_flag:
                for param in self.model.model_encoder.parameters():
                    param.requires_grad = False
        else:
            for param in self.model.parameters():
                param.requires_grad = False
            if not self.fix_H_flag:
                for param in self.model.h_learned_lower_dimension.parameters():
                    param.requires_grad = True
            if not self.fix_encoder_flag:
                for param in self.model.model_encoder.parameters():
                    param.requires_grad = True


        ################## CV space declarations ##########################
        self.N_CV = n_CV
        ## tracking optimal validation value ##
        self.MSE_cv_dB_opt = 1000
        self.MSE_cv_idx_opt = 0

        ## linear MSE for each batch (samples in batch) in CV ##
        MSE_cv_linear_batch = torch.empty([self.N_CV])
        ## linear MSE for each epoch (all samples in CV) ##
        self.MSE_cv_linear_epoch = torch.empty([self.N_Epochs])
        ## log MSE for each epoch (all samples in CV) ##
        self.MSE_cv_dB_epoch = torch.empty([self.N_Epochs])

        ## linear MSE for each batch (samples in batch) in CV for encoder##
        MSE_cv_linear_batch_encoder = torch.empty([self.N_CV])
        ## linear MSE for each epoch (all samples in CV) for encoder##
        self.MSE_cv_linear_epoch_encoder = torch.empty([self.N_Epochs])
        ## log MSE for each epoch (all samples in CV) for encoder##
        self.MSE_cv_dB_epoch_encoder = torch.empty([self.N_Epochs])
        ####################################################

        ##############
        ### Epochs ###
        ##############

        Train_loss_list=[]
        Val_loss_list = []
        for ti in range(0, self.N_Epochs):
            t = time.time()
            #################################
            ### Validation Sequence Batch ###
            #################################
            self.model.eval()
            ################ running on each sample ########
            for j in range(0, self.N_CV):
                self.model.InitSequence(self.ssModel.m1x_0, self.ssModel.T)
                y_cv = cv_input[j, :, :]
                x_out_cv, z_encoder_output_cv = self.get_seq_knet_output(y_cv, cv_target[j, :, :],prior_flag)
                # Compute Training Loss
                MSE_cv_linear_batch[j] = self.loss_fn(x_out_cv, cv_target[j, :, :]).detach()
                MSE_cv_linear_batch_encoder[j] = self.loss_fn(z_encoder_output_cv.float(), cv_target[j, :, :].float())
                if MSE_cv_linear_batch[j].isnan() == True:
                    Itay = 29
                    MSE_cv_linear_batch[j] = 1
                    print("**** we have nan value ****")
                    #break
                if j == 4:
                    print("encoder output {} x state {} kalman output {}".format(z_encoder_output_cv[:, 25], cv_target[j, :, 25],x_out_cv[:, 25]))
            # Average
            self.MSE_cv_linear_epoch[ti] = torch.mean(MSE_cv_linear_batch)
            self.MSE_cv_dB_epoch[ti] = 10 * torch.log10(self.MSE_cv_linear_epoch[ti])

            self.MSE_cv_linear_epoch_encoder[ti] = torch.mean(MSE_cv_linear_batch_encoder)
            self.MSE_cv_dB_epoch_encoder[ti] = 10 * torch.log10(self.MSE_cv_linear_epoch_encoder[ti])

            # saving if better than optimal weights
            if (self.MSE_cv_dB_epoch[ti] < self.MSE_cv_dB_opt):
                self.MSE_cv_dB_opt = self.MSE_cv_dB_epoch[ti]
                self.MSE_cv_idx_opt = ti
                torch.save(self.model, self.modelFileName)

            ###############################
            ### Training Sequence Batch ###
            ###############################

            ################## train space declarations ##########################
            self.N_E = n_Examples
            ## linear MSE for each batch (samples in batch) in train ##
            MSE_train_linear_batch = torch.empty([self.N_B])
            ## linear MSE for each epoch (all samples in train) ##
            self.MSE_train_linear_epoch = torch.empty([self.N_Epochs])
            ## log MSE for each epoch (all samples in train) ##
            self.MSE_train_dB_epoch = torch.empty([self.N_Epochs])

            ## linear MSE for each batch (samples in batch) in train for encoder##
            MSE_train_linear_batch_encoder = torch.empty([self.N_B])
            ## linear MSE for each epoch (all samples in train) encoder##
            self.MSE_train_linear_epoch_encoder = torch.empty([self.N_Epochs])
            ## log MSE for each epoch (all samples in train) encoder##
            self.MSE_train_dB_epoch_encoder = torch.empty([self.N_Epochs])
            ########################################################################

            # Training Mode
            #self.model.train()
            # Init Hidden State
            self.model.init_hidden()
            Batch_Optimizing_LOSS_sum = 0
            Batch_Optimizing_LOSS_sum_Encoder = 0

            for j in range(0, self.N_B):
                print(j)
                n_e = random.randint(0, self.N_E - 1)
                self.model.InitSequence(self.ssModel.m1x_0, self.ssModel.T)
                y_training = train_input[n_e, :, :]
                x_out_training, z_encoder_output_train = self.get_seq_knet_output(y_training, train_target[n_e, :, :], prior_flag)
                # Compute Training Loss
                LOSS = self.loss_fn(x_out_training.float(), train_target[n_e, :, :].float())
                LOSS_Encoder = self.loss_fn(z_encoder_output_train.float(), train_target[n_e, :, :].float())
                MSE_train_linear_batch[j] = LOSS.detach()
                MSE_train_linear_batch_encoder[j] = LOSS_Encoder.detach()
                # if MSE_train_linear_batch[j].isnan() == True:
                #     Itay = 29
                #     MSE_train_linear_batch[j] = 1
                #     LOSS=1
                # if MSE_train_linear_batch_encoder[j].isnan() == True:
                #     Itay = 29
                #     MSE_train_linear_batch_encoder[j] = 1
                #     LOSS_Encoder=1
                Batch_Optimizing_LOSS_sum = Batch_Optimizing_LOSS_sum + LOSS
                Batch_Optimizing_LOSS_sum_Encoder = Batch_Optimizing_LOSS_sum_Encoder + LOSS_Encoder
            # Average
            self.MSE_train_linear_epoch[ti] = torch.mean(MSE_train_linear_batch)
            self.MSE_train_dB_epoch[ti] = 10 * torch.log10(self.MSE_train_linear_epoch[ti]).cpu().detach().numpy()

            self.MSE_train_linear_epoch_encoder[ti] = torch.mean(MSE_train_linear_batch_encoder)
            self.MSE_train_dB_epoch_encoder[ti] = 10 * torch.log10(self.MSE_train_linear_epoch_encoder[ti]).cpu().detach().numpy()

            ##################
            ### Optimizing ###
            ##################
            # if ti%2==0:
            self.optimizer.zero_grad()
            Batch_Optimizing_LOSS_mean = Batch_Optimizing_LOSS_sum / self.N_B
            Batch_Optimizing_LOSS_mean.backward()
            self.optimizer.step()
            # else:
            #     self.optimizer.zero_grad()
            #     Batch_Optimizing_LOSS_encoder_mean = Batch_Optimizing_LOSS_sum_Encoder/self.N_B
            #     Batch_Optimizing_LOSS_encoder_mean.backward()
            #     self.optimizer.step()
            ########################
            ### Training Summary ###
            ########################
            print(ti, "MSE Training :", self.MSE_train_dB_epoch[ti].cpu().detach().numpy(), "[dB]", "MSE Validation :", self.MSE_cv_dB_epoch[ti].cpu().detach().numpy(),"[dB]","timing ", time.time() - t)
            print(ti, "Enc Training :", self.MSE_train_dB_epoch_encoder[ti].cpu().detach().numpy(), "[dB]", "Enc Validation :", self.MSE_cv_dB_epoch_encoder[ti].cpu().detach().numpy(),"[dB]")
            Train_loss_list.append(self.MSE_train_dB_epoch[ti].cpu().detach().numpy())
            Val_loss_list.append(self.MSE_cv_dB_epoch[ti].cpu().detach().numpy())
            #self.check_tracking(ti, x_out_training, train_target, n_e, z_encoder_output_train, folder_learning_path)
            # if (self.MSE_cv_dB_epoch[ti]>5):
            #     print("configuration is not good enough")
            #     break
        #self.print_process(Val_loss_list, Train_loss_list, title)
        print("Optimal idx:", self.MSE_cv_idx_opt, "Optimal :", self.MSE_cv_dB_opt, "[dB]")

    # def print_process(self, val_loss_list, train_loss_list, title):
    #     fig = plt.figure(figsize=(10, 7))
    #     fig.add_subplot(1, 1, 1)
    #     plt.plot(train_loss_list, label='train')
    #     plt.plot(val_loss_list, label='val')
    #     plt.title("Loss of {}".format(title))
    #     plt.legend()
    #     plt.savefig(self.Learning_process_folderName +'Learning_curve.jpeg')

    def NNTest(self, n_Test, test_input, test_target,prior_flag):
        length_seq = test_input.shape[2]
        self.MSE_test_linear_arr = torch.empty([n_Test])
        self.MSE_test_linear_arr_encoder = torch.empty([n_Test])
        # MSE LOSS Function
        loss_fn = nn.MSELoss(reduction='mean')
        #self.model = torch.load(self.modelFileName)
        self.model.eval()
        torch.no_grad()
        start = time.time()
        x_out_test_all = torch.empty(n_Test, self.ssModel.m, self.ssModel.T_test)
        encoder_test_all = torch.empty(n_Test, self.ssModel.m, self.ssModel.T_test)

        for j in range(0, n_Test): #running on each sample (trajectory)
            self.model.InitSequence(self.ssModel.m1x_0, self.ssModel.T_test)
            y_mdl_tst = test_input[j, :, :] #taking the j observation sample (trajectory)
            x_out, z_encoder_output = self.get_seq_knet_output(y_mdl_tst,test_target[j,:,:],prior_flag)
            #x_out = torch.empty(self.ssModel.m, length_seq)
            #z_encoder_output = torch.empty(self.ssModel.m, length_seq)
            #test_input1=test_input[:50,:,:]
            #test_target1=test_target[:50,:,:]
            #for t in range(0, length_seq):
            #    AE_input = y_mdl_tst[:,:,t].reshape(1, y_size, y_size)
            #    y_decoaded = self.model.model_encoder(AE_input.double()).squeeze()
            #    encoder_test_all[:,:,t] = y_decoaded
            #    x_out_test_all[:,:,t]= self.model(y_decoaded)
                # Compute Training Loss
            x_out_test_all[j,:,:] = x_out
            encoder_test_all[j,:,:] = z_encoder_output
            self.MSE_test_linear_arr[j] = loss_fn(x_out, test_target[j, :, :]).detach()
            self.MSE_test_linear_arr_encoder[j]= loss_fn(z_encoder_output, test_target[j, :, :]).detach()
            print(j)
        infer_time = time.time() - start
        ####### KNLS #####################
        # Average
        self.MSE_test_linear_avg = torch.mean(self.MSE_test_linear_arr)
        self.MSE_test_dB_avg = 10 * torch.log10(self.MSE_test_linear_avg)
        # Standard deviation
        self.MSE_test_dB_std = torch.std(self.MSE_test_linear_arr, unbiased=True)
        self.MSE_test_dB_std = 10 * torch.log10(self.MSE_test_dB_std)
        # Print MSE Cross Validation
        str = self.modelName + "-" + "MSE Test:"
        print(str, self.MSE_test_dB_avg, "[dB]")
        str = self.modelName + "-" + "STD Test:"
        print(str, self.MSE_test_dB_std, "[dB]")
        # Print Run Time
        print("Inference Time:", infer_time)
        # histogram of KFLS
        #plt.hist(self.MSE_test_linear_arr, 10)
        #plt.show()

        ####### Encoder #####################
        # Average
        self.MSE_test_linear_avg_encoder = torch.mean(self.MSE_test_linear_arr_encoder)
        self.MSE_test_dB_avg_encoder = 10 * torch.log10(self.MSE_test_linear_avg_encoder)
        # Standard deviation
        self.MSE_test_dB_std_encoder = torch.std(self.MSE_test_linear_arr_encoder, unbiased=True)
        self.MSE_test_dB_std_encoder = 10 * torch.log10(self.MSE_test_dB_std_encoder)
        # Print MSE Cross Validation
        str = "Only Encoder" + "-" + "MSE Test:"
        print(str, self.MSE_test_dB_avg_encoder, "[dB]")
        str = "Only Encoder" + "-" + "STD Test:"
        print(str, self.MSE_test_dB_std_encoder, "[dB]")
        # histogram of encoder
        #plt.hist(self.MSE_test_linear_arr_encoder, 10)
        #plt.show()

        return [encoder_test_all, x_out_test_all]

    def PlotTrain_KF(self, MSE_KF_linear_arr, MSE_KF_dB_avg):

        self.Plot = Plot(self.folderName, self.modelName)

        self.Plot.NNPlot_epochs(self.N_Epochs, MSE_KF_dB_avg, self.MSE_test_dB_avg, self.MSE_cv_dB_epoch, self.MSE_train_dB_epoch)

        self.Plot.NNPlot_Hist(MSE_KF_linear_arr, self.MSE_test_linear_arr)