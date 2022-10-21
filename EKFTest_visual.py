## EKFTest_visual ##
import torch.nn as nn
import torch
from EKF_visual import ExtendedKalmanFilter
import matplotlib.pyplot as plt
import time

def EKFTest(SysModel, test_input, test_target,Lorenz_data_flag, EKF_with_encoder_flag, model_encoder_trained, modelKnowledge='full', allStates=True):
    N_T = test_target.size()[0]
    # LOSS
    loss_fn = nn.MSELoss(reduction='mean')

    # MSE [Linear]
    MSE_EKF_linear_arr = torch.empty(N_T)
    MSE_encoder_linear_arr = torch.empty(N_T)

    EKF = ExtendedKalmanFilter(SysModel,Lorenz_data_flag,EKF_with_encoder_flag, model_encoder_trained,modelKnowledge)
    EKF.InitSequence_EKF(SysModel.m1x_0, SysModel.m2x_0)

    KG_array = torch.zeros_like(EKF.KG_array)
    EKF_out = torch.empty([N_T, SysModel.m, SysModel.T_test])

    for j in range(0, N_T):
        #t_start = time.time()
        print(j)
        EKF.GenerateSequence(test_input[j, :, :], test_target[j, :, :], EKF.T_test) #j sample (all trajectory is getting to EKF pipeline)
        #print('time of sample {} is {}'.format(j, time.time()-t_start))
        #test_input[j, :, :], test_target[j, :, :]
        if (allStates):
            MSE_EKF_linear_arr[j] = loss_fn(EKF.x, test_target[j, :, :]).item()
            MSE_encoder_linear_arr[j]= loss_fn(EKF.encoder_output, test_target[j, :, :]).item()
            if EKF_with_encoder_flag:
               print("loss EKF with encoder {}".format(10 * torch.log10(MSE_EKF_linear_arr[j])))
               print("loss encoder {}".format(10 * torch.log10(MSE_encoder_linear_arr[j])))
            else:
               print("loss EKF of sample {} is {}".format(j,10 * torch.log10(MSE_EKF_linear_arr[j])))
        else:
            loc = torch.tensor([True, False, True, False])
            MSE_EKF_linear_arr[j] = loss_fn(EKF.x[loc, :], test_target[j, :, :]).item()
        KG_array = torch.add(EKF.KG_array, KG_array)
        EKF_out[j, :, :] = EKF.x
        if MSE_EKF_linear_arr[j].isnan()==True:
            print('we have nan result in sample j')
            MSE_EKF_linear_arr[j]=1
            break
        if MSE_EKF_linear_arr[j] > 0.4:
            print('MSE is {} so q={} value is not good'.format(MSE_EKF_linear_arr[j] ,EKF.givenQ))
            break
        #Check_Changes_tracking_EKF(EKF, test_target, j)

    # Average KG_array over Test Examples
    KG_array /= N_T

    # Average
    MSE_EKF_linear_avg = torch.mean(MSE_EKF_linear_arr)
    MSE_EKF_dB_avg = 10 * torch.log10(MSE_EKF_linear_avg)
    # Standard deviation
    #MSE_EKF_linear_std = torch.std(MSE_EKF_linear_arr,unbiased=True)
    #MSE_EKF_dB_std = 10 * torch.log10(MSE_EKF_linear_std)
    # Print MSE Cross Validation
    #str = "EKF with encoder" + "-" + "MSE Test:"
    #print(str, MSE_EKF_dB_avg, "[dB]")
    #str = "EKF with encoder" + "-" + "STD Test:"
    #print(str, MSE_EKF_dB_std, "[dB]")

    # histogram of EKF with encoder
    #plt.hist(MSE_EKF_linear_arr, 10)
    #plt.show()

    if EKF_with_encoder_flag:
        print("Extended Kalman Filter with encoder - Test MSE LOSS:", MSE_EKF_dB_avg, "[dB]")
    else:
        print("Extended Kalman Filter - Test MSE LOSS:", MSE_EKF_dB_avg, "[dB]")
    return [MSE_EKF_linear_arr, MSE_EKF_linear_avg, MSE_EKF_dB_avg, KG_array, EKF_out, test_target]


def EKFTrain(SysModel, model_encoder_trained,  train_input, train_target, cv_input, cv_target,EKF_with_encoder_flag, title, folder_learning_path, modelKnowledge='full', allStates=True):
    N_e = train_input.size()[0]
    # LOSS
    loss_fn = nn.MSELoss(reduction='mean')
    # MSE [Linear] and output declare space
    MSE_EKF_linear_arr = torch.empty(N_e)
    EKF_out = torch.empty([N_e, SysModel.m, SysModel.T])
    # EKF modul
    EKF = ExtendedKalmanFilter(SysModel, EKF_with_encoder_flag, model_encoder_trained,modelKnowledge)
    EKF.InitSequence_EKF(SysModel.m1x_0, SysModel.m2x_0)
    ## optimaization
    model_encoder_trained.train()
    for j in range(0, N_e):
        print(j)
        EKF.GenerateSequence(train_input[j, :, :],train_target[j, :, :], EKF.T)
        if (allStates):
            MSE_EKF_linear_arr[j] = loss_fn(EKF.x, train_target[j, :, :]).item()
        else:
            loc = torch.tensor([True, False, True, False])
            MSE_EKF_linear_arr[j] = loss_fn(EKF.x[loc, :], train_target[j, :, :]).item()
        EKF_out[j, :, :] = EKF.x
        if EKF_with_encoder_flag:
            EKF.optimizer.zero_grad()
            loss_for_training = EKF.loss_EKF_with_encoser_for_training(EKF_out[j, :, :], train_target[j, :, :])
            loss_for_training.backward(retain_graph=True)
            EKF.optimizer.step()
        MSE_EKF_linear_avg = torch.mean(MSE_EKF_linear_arr)
        MSE_EKF_dB_avg = 10 * torch.log10(loss_for_training)
        print("sample:", j," Extended Kalman Filter - Train MSE LOSS:", MSE_EKF_dB_avg, "[dB]")
    return [MSE_EKF_linear_arr, MSE_EKF_linear_avg, MSE_EKF_dB_avg, EKF_out, test_target]

def Check_Changes_tracking_EKF(EKF,test_target,j):
    fig, (ax1, ax2, ax3) = plt.subplots(3)
    fig.suptitle("components Estimation")
    EKF_with_encoder_0 = EKF.x[0, :].cpu().detach().numpy()
    GT_0 = test_target[j, 0, :].cpu()
    Enc_0 = EKF.encoder_output[0, :].cpu().detach().numpy()
    axis = list(range(GT_0.shape[0]))
    ax1.plot(axis, GT_0)
    ax1.plot(axis, EKF_with_encoder_0)
    ax1.plot(axis, Enc_0)
    ax1.legend(["GT", "EKF_with_encoder", "Encoder"])

    EKF_with_encoder_1 = EKF.x[1, :].cpu().detach().numpy()
    GT_1 = test_target[j, 1, :].cpu()
    Enc_1 = EKF.encoder_output[1, :].cpu().detach().numpy()
    axis = list(range(GT_1.shape[0]))
    ax2.plot(axis, GT_1)
    ax2.plot(axis, EKF_with_encoder_1)
    ax2.plot(axis, Enc_1)
    ax2.legend(["GT", "EKF_with_encoder", "Encoder"])

    EKF_with_encoder_2 = EKF.x[2, :].cpu().detach().numpy()
    GT_2 = test_target[j, 2, :].cpu()
    Enc_2 = EKF.encoder_output[2, :].cpu().detach().numpy()
    axis = list(range(GT_2.shape[0]))
    ax3.plot(axis, GT_2)
    ax3.plot(axis, EKF_with_encoder_2)
    ax3.plot(axis, Enc_2)
    ax3.legend(["GT", "EKF_with_encoder", "Encoder"])
    fig.show()

    fig, ax = plt.subplots(2, 2, figsize=(10, 10), subplot_kw=dict(projection='3d'))
    fig.suptitle("components 3D comparison")
    EKF_with_encoder_3D = ax[0, 0].plot(EKF_with_encoder_0, EKF_with_encoder_1, EKF_with_encoder_2)
    ax[0, 0].set_title("EKF_with_encoder")
    Enc_3D = ax[0, 1].plot(Enc_0, Enc_1, Enc_2)
    ax[0, 1].set_title("Encoder")
    GT_3D = ax[1, 0].plot(GT_0, GT_1, GT_2)
    ax[1, 0].set_title("GT")
    fig.show()