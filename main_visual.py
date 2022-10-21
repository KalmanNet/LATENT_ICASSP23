## main_visual ##
from Extended_sysmdl_visual import SystemModel
from Extended_data_visual import DataGen,DataLoader_GPU
from EKFTest_visual import EKFTest
from Pipeline_KF_visual_gpu import Pipeline_KF
from KalmanNet_nn_visual_new_architecture_gpu import KalmanNetNN, in_mult, out_mult
from config import *
from main_AE import inference_with_prior, initialize_data_AE, d, test_epoch
import math, time, random

####################### running configurations ##############
strTime, sinerio, h_latent_space, model_encoder_trained, data_configuration_name, data_name, folder_KNet_name, folder_EKF_results, folder_simulations,directory_for_trained_KGain, H_matrix_for_visual = config_dimentions_and_dierctories()
dev = define_dev() # define if working on CPU or GPU with dev
##############################################################

### Design Models #####################################################################
sys_model = SystemModel(F, given_q, real_q, H, h, given_r, real_r, T, T_test, m, n, data_name, EKF_with_encoder_flag)
sys_model.InitSequence(m1x_0, m2x_0)
print("1. create system model")
############################################################################################

###### data generation / loading #############################################################
if data_gen_flag:
    print("2. Start Data Gen")
    [train_input, train_target, cv_input, cv_target, test_input, test_target] = DataGen(sys_model, folder_simulations + strTime +data_configuration_name, T, T_test,N_E, N_CV, N_T, randomInit=False)  # taking time
else:
    print("2. Data Load")
    [train_input, train_target, cv_input, cv_target, test_input, test_target] = DataLoader_GPU(folder_simulations +data_configuration_name)
print("trainset size: x {} y {}".format(train_target.size(),train_input.size()))
print("cvset size: x {} y {}".format(cv_target.size(), cv_input.size()))
print("testset size: x {} y {}".format(test_target.size(), test_input.size()))
###############################################################################################

### Evaluate Extended Kalman Filter ###############
for q in [given_q]:
    # 0.01, 0.1,0.5, 1, 2, 5, 10
    for r in [given_r]:
        #0.2, 0.3, 0.5, 0.6, 0.7
        #print('q = {} r = {} '.format(q, r))
        if EKF_with_encoder_flag:
            sys_model.givenQ = q * q * torch.eye(d)
            sys_model.givenR = r * r * torch.eye(d)
        else:
            sys_model.givenQ = q * q * torch.eye(m)
            sys_model.givenR = r * r * torch.eye(n)
        if evaluate_EKF_flag:
            if EKF_with_encoder_flag:
                print("3. Evaluate Extended Kalman Filter with encoder")
                start=time.time()
                [MSE_EKF_linear_arr, MSE_EKF_linear_avg, MSE_EKF_dB_avg_with_encoder, EKF_KG_array, EKF_out_with_encoder, GT_test] = EKFTest(sys_model, test_input, test_target, Lorenz_data_flag, EKF_with_encoder_flag, model_encoder_trained)
                print("Inference Time:", time.time() - start)
                torch.save([EKF_out_with_encoder, MSE_EKF_dB_avg_with_encoder, GT_test], folder_EKF_results+'/EKF_fix_encoder_r_{}_q_{}_{}.pt'.format(given_r,given_q, dataset))
            else:
                print("3. Evaluate Extended Kalman Filter")
                start = time.time()
                [MSE_EKF_linear_arr, MSE_EKF_linear_avg, MSE_EKF_dB_avg, EKF_KG_array, EKF_out, GT_test] = EKFTest(sys_model, test_input, test_target, Lorenz_data_flag,EKF_with_encoder_flag, model_encoder_trained)
                print("Inference Time:", time.time() - start)
                torch.save([EKF_out, MSE_EKF_dB_avg, GT_test], folder_EKF_results+'/EKF_r_{}_q_{}.pt'.format(given_r,given_q))
###################################################################################################################

############## checking only encoder loss (not depand on seed) ####################################
train_loader,valid_loader, test_loader, test_input, test_target= initialize_data_AE(folder_simulations +data_configuration_name, batch_size,prior_r)
if prior_flag:
    test_loss = inference_with_prior(model_encoder_trained, test_input, test_target)
else:
    test_loss = test_epoch(model_encoder_trained, test_loader, torch.nn.MSELoss(), batch_size, prior_flag)
print("4. Test loss of encoder is: {} dB".format(10 * math.log10(test_loss)))
####################################################################################################

############# set seed  ###################################################################
num = 0
torch.manual_seed(num)
random.seed(num)
print("set seed to be {}".format(num))
############################################################################################

################## KNet Pipeline #########################################################
print("5. create KNet pipeline instance")
KNet_Pipeline = Pipeline_KF(strTime,sinerio, folder_KNet_name, data_name,  fix_H_flag, fix_encoder_flag, Lorenz_data_flag, real_r)
KNet_Pipeline.setssModel(sys_model)
############################################################################################

################## K Gain model ##########################################################
if load_KGain_trained_flag:
    print("6. Load trained KGain")
    KNet_Pipeline.model = torch.load(directory_for_trained_KGain).double()
    #KNet_Pipeline.model.h_Sigma = torch.randn(1, 100, 9)
else:
    print("6. create KGain instance")
    KGain_model = KalmanNetNN().double()
    KGain_model.Build(sys_model, h_latent_space, model_encoder_trained, H_matrix_for_visual, fix_H_flag,fix_encoder_flag)
    KNet_Pipeline.setModel(KGain_model.double())
############################################################################################

################## training ################################################################
if flag_Train:
    print("7. start  KNet pipeline training over training set")
    KNet_Pipeline.setTrainingParams(n_Epochs=epoches, n_Batch=batch_size, learningRate=lr_kalman, weightDecay=wd_kalman)
    title="LR: {} Weight Decay: {} model complexity in_mult = {} out_mult = {}".format(lr_kalman,wd_kalman,in_mult, out_mult )
    print_weights( KNet_Pipeline, 'before')
    KNet_Pipeline.NNTrain(N_E, train_input, train_target, N_CV, cv_input, cv_target, title, prior_flag )
    print_weights(KNet_Pipeline, 'after')
    KNet_Pipeline.save()
#############################################################################################

##################### Compare models on Test Set ##########################################
print("start  KNet pipeline inference over test set")
[encoder_test, KNet_test] = KNet_Pipeline.NNTest(N_T, test_input, test_target,prior_flag)
pytorch_total_params = sum(p.numel() for p in KNet_Pipeline.model.parameters() if p.requires_grad)
print("Knet Pipeline include {} trainable parameters".format(pytorch_total_params))

######################## convert to numpy for rkn code ##############################
# x=test_target[68]
# fig = plt.figure()
# ax = fig.gca(projection="3d")
# ax.plot(x[0,:], x[1,:], x[2,:])
# plt.draw()
# plt.show()

# train_input_reshaped=train_input.transpose(1,2).reshape((1000,200,28,28,1)).numpy()
# test_input_reshaped=test_input.transpose(1,2).reshape((100,200,28,28,1)).numpy()
# cv_input_reshaped=cv_input.transpose(1,2).reshape((100,200,28,28,1)).numpy()
# train_target_reshaped=train_target.transpose(1,2).numpy()
# test_target_reshaped=test_target.transpose(1,2).numpy()
# cv_target_reshaped=cv_target.transpose(1,2).numpy()
# print("trainset size: x {} y {}".format(train_target_reshaped.shape,train_input_reshaped.shape))
# print("cvset size: x {} y {}".format(cv_target_reshaped.shape, cv_input_reshaped.shape))
# print("testset size: x {} y {}".format(test_target_reshaped.shape, test_input_reshaped.shape))
# np.save('./Simulations/new_lorenz_for_rkn_T=200_q_{}_r_{}'.format(real_q,real_r), [train_input_reshaped, train_target_reshaped, cv_input_reshaped, cv_target_reshaped, test_input_reshaped, test_target_reshaped])
# [train_input_reshaped, train_target_reshaped, cv_input_reshaped, cv_target_reshaped, test_input_reshaped, test_target_reshaped]=np.load('./Simulations/new_lorenz_for_rkn_T=200_q_{}_r_{}.npy'.format(real_q,real_r),allow_pickle=True)
# Itay=29

###########################  plot results ##################################
# ######## baseline ###############
# results_encoder = [7.8,1.03,-3.1,-7.97]
# results_encoder_with_prior_r_4 = [6.53,1.62,-3.43,-7.81]
# results_EKF = [6.89, 0.753,-3.615,-8.01]
# results_rkn = [4, -2,-4.7,-8.3]
# results_KNet_fixed_Encoder = [4.64,-1.2,-4.6,-8.23]
# results_KNet_learned_Encoder = [3.1,-2.53,-4.97,-8.32]
# results_KNet_learned_Encoder_with_prior = [3.87,-1.27,-4.4,-8.6]
# x_axis = [-20,-12,-6,0]
# fig, ax = plt.subplots()
# ax.plot(x_axis,results_encoder, marker='o', label='Encoder', color='red' )
# #ax.plot(x_axis,results_encoder_with_prior_r_4, marker='^', label='Encoder with prior' )
# ax.plot(x_axis,results_EKF, marker='*', label='EKF with Encoder', color='orange')
# ax.plot(x_axis,results_rkn, marker='d', label='RKN', color='purple')
# ax.plot(x_axis,results_KNet_fixed_Encoder, marker='x', label='KalmanNet with fixed Encoder')
# ax.plot(x_axis,results_KNet_learned_Encoder, marker='s', label='Learned Kalman Filter in Latent Space',color='green')
# #ax.plot(x_axis,results_KNet_learned_Encoder_with_prior, marker='x', label='KNet with learned Encoder + prior')
# plt.title('MSE vs 1/r^2 over Baseline T_train=T_test=200')
# plt.xlabel(r'$\frac{1}{r^2}$ [dB]')
# plt.ylabel('MSE [dB]')
# plt.legend()
# plt.grid(True)
# plt.show()
#
# ################## long #############################
# results_encoder = [7.17, 0.88, -4.11, -9.03]
# results_encoder_with_prior_r_4 = [6.84,1.86,-3.43,-7.81]
# results_EKF = [6.213, 0.44,-4.38,-9.06]
# results_rkn = [7.8, 1.55,-3.136,-9]
# results_KNet_fixed_Encoder = [4.31,-0.53,-4.74,-9.4,]
# results_KNet_learned_Encoder = [3.52,-1.35,-5,-9.12]
# results_KNet_learned_Encoder_with_prior = [4.36,-0.146,-4.36,-9.06]
# fig, ax = plt.subplots()
# ax.plot(x_axis,results_encoder, marker='o', label='Encoder', color='red' )
# ax.plot(x_axis,results_EKF, marker='*', label='EKF with Encoder', color='orange')
# ax.plot(x_axis,results_rkn, marker='d', label='RKN', color='purple')
# ax.plot(x_axis,results_KNet_fixed_Encoder, marker='x', label='KalmanNet with fixed Encoder')
# ax.plot(x_axis,results_KNet_learned_Encoder, marker='s', label='Learned Kalman Filter in Latent Space',color='green')
# plt.title(r'MSE vs $\frac{1}{r^2}$ over long trajectories T_train=200 T_test=2000')
# plt.xlabel(r'$\frac{1}{r^2}$ [dB]')
# plt.ylabel('MSE [dB]')
# plt.legend()
# plt.grid(True)
# plt.show()

############################ plot predictions of different algorithms ###################
# sample=44
# initial_encoder = torch.empty(3, 2000)
# trajectory = test_input[sample,:,:]
# for y in range(2000):
#     initial_encoder[:,y] = model_encoder_trained(trajectory[:,y].reshape(1,1,28,28))
#
# #initial_encoder = initial_encoder.detach().numpy()
# fig = plt.figure()
# ax = fig.gca(projection="3d")
# #plt.axis('off')
# plt.grid(True)
# ax.plot(initial_encoder[0, :], initial_encoder[1, :], initial_encoder[2, :], color='red')
# ax.set_yticklabels([])
# ax.set_xticklabels([])
# ax.set_zticklabels([])
# plt.draw()
# plt.show()
#
# fig = plt.figure()
# ax = fig.gca(projection="3d")
# plt.grid(True)
# #plt.axis('off')
# ax.plot(test_target[sample,0,:], test_target[sample,1,:], test_target[sample,2,:], color='black')
# ax.set_yticklabels([])
# ax.set_xticklabels([])
# ax.set_zticklabels([])
# plt.draw()
# plt.show()
#
# #encoder_test = encoder_test.detach().numpy()
# fig = plt.figure()
# ax = fig.gca(projection="3d")
# plt.grid(True)
# #plt.axis('off')
# ax.plot(encoder_test[sample,0,:], encoder_test[sample,1,:], encoder_test[sample,2,:], color='purple')
# ax.set_yticklabels([])
# ax.set_xticklabels([])
# ax.set_zticklabels([])
# plt.draw()
# plt.show()
#
# #KNet_test = KNet_test.detach().numpy()
# fig = plt.figure()
# ax = fig.gca(projection="3d")
# plt.grid(True)
# #plt.axis('off')
# ax.plot(KNet_test[sample,0,:], KNet_test[sample,1,:], KNet_test[sample,2,:], color='green')
# ax.set_yticklabels([])
# ax.set_xticklabels([])
# ax.set_zticklabels([])
# plt.draw()
# plt.show()
#
# for t in range(2000):
#     if t%200==0:
#         fig = plt.figure()
#         imgplot = plt.imshow(trajectory[:,t].reshape(28,28))
#         plt.colorbar()
#         plt.show()
##############################################################################################
