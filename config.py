##########  config  ##########
import torch
import yaml
from datetime import datetime # getting current time
from model_lorenz_visual import h_latent_space, H, F, h, encoded_dimention, b, m, n, y_size, m1x_0, m2x_0, get_h_derivative, eps
from main_AE import Encoder_conv, Encoder_conv_with_prior
import argparse


def get_arguments():
  parser = argparse.ArgumentParser()
  # Data
  parser.add_argument('--dataset_configuration', default=3, help="1 - Ttrain=Ttest=200, 2 - Ttrain=200 Ttest=200 but decimated, 3 - Ttrain=200 Ttest=2000")
  parser.add_argument('--Lorenz_data_flag', action='store_true', default=1, help="True - Load Lorenz Dataset. False - Load Syntetic Dataset")
  parser.add_argument('--data_gen_flag', action='store_true', default=0, help="True - Generating Dataset. False - Loading Dataset")
  parser.add_argument('--flag_H_Identity', action='store_true', default=1, help="True - Observation Matrix is I. False - Observation Matrix is spesific")
  parser.add_argument('--fix_H_flag', action='store_true', default=0, help="True - fix Observation Matrix. False - Learning Observation Matrix")
  # Knet
  parser.add_argument('--prior_flag', action='store_true', default=0, help="True - Encoder with prior. False - Encoder without prior")
  parser.add_argument('--fix_encoder_flag', action='store_true', default=0, help="True - Encoder is fixed. False - Encoder is trainable")
  parser.add_argument('--load_KGain_trained_flag', action='store_true', default=1, help="True - loading trained KGain model. False - KGain model start from scratch")
  parser.add_argument('--flag_Train', action='store_true', default=0, help="True - Training full pipeline. False - Only inference")
  # EKF
  parser.add_argument('--evaluate_EKF_flag', action='store_true', default=0, help="True - Evaluate EKF results. False - Load EKF results")
  parser.add_argument('--EKF_with_encoder_flag', action='store_true', default=0, help="True - EKF is working with trained encoder. False - Only EKF")
  parser.add_argument('--EKF_with_encoder_flag_Train', action='store_true', default=0, help="True - Training encoder in EKF process. False - fix encoder")
  # directories
  param_dict = yaml_configuration('./configurations/config_file.yaml')
  return parser, param_dict

def yaml_configuration(path):
    with open(path, "r") as stream:
      try:
        param_dict = yaml.safe_load(stream)
      except yaml.YAMLError as exc:
        print(exc)
    return param_dict

############ parser ######################################
parser, param_dict = get_arguments()
## flags
opt = parser.parse_args()
dataset_configuration = opt.dataset_configuration
Lorenz_data_flag = opt.Lorenz_data_flag
data_gen_flag = opt.data_gen_flag
flag_H_Identity = opt.flag_H_Identity
fix_H_flag = opt.fix_H_flag

prior_flag = opt.prior_flag
fix_encoder_flag = opt.fix_encoder_flag
load_KGain_trained_flag = opt.load_KGain_trained_flag
flag_Train = opt.flag_Train

evaluate_EKF_flag = opt.evaluate_EKF_flag
EKF_with_encoder_flag= opt.EKF_with_encoder_flag
EKF_with_encoder_flag_Train = opt.EKF_with_encoder_flag_Train

############ Hyper Parameters ##################
lr_kalman = param_dict.get("lr_kalman")
wd_kalman = param_dict.get("wd_kalman")
batch_size = param_dict.get("batch_size")
epoches = param_dict.get("epoches")
########### Noise statistics for EKF #############
real_r = param_dict.get("real_r")
real_q = param_dict.get("real_q")
prior_r = param_dict.get("prior_r")

############## data ##################
if dataset_configuration==1:
  decimation_flag = False
  dataset = 'Ttest=200 T=200'
  ########### Size of DataSet ##################
  N_E = 1000  # Number of Training Examples
  N_CV = 100  # Number of Cross Validation Examples
  N_T = 100  # Number of Test Examples
  T = 200  # Sequence Length Train
  T_test = 200  # Sequence Length Test
  # EKF with encoder given noise values
  given_r = 0.5
  given_q = 3.7
if dataset_configuration==2:
  decimation_flag = True
  dataset ='Ttest=200 decimated T=200'
  ########### Size of DataSet ##################
  N_E = 1000  # Number of Training Examples
  N_CV = 100  # Number of Cross Validation Examples
  N_T = 100  # Number of Test Examples
  T = 200  # Sequence Length Train
  T_test = 200  # Sequence Length Test
  # EKF with encoder given noise values
  given_r = 5.09
  given_q = 15
if dataset_configuration == 3:
  decimation_flag = False
  dataset = 'Ttest=2000 T=200'
  ########### Size of DataSet ##################
  N_E = 2  # Number of Training Examples
  N_CV = 2  # Number of Cross Validation Examples
  N_T = 100  # Number of Test Examples
  T = 2000  # Sequence Length Train
  T_test = 2000  # Sequence Length Test
  # EKF with encoder given noise values
  given_r = 0.046
  given_q = 0.5

def define_dev():
  if torch.cuda.is_available():
    dev = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    print("Running on the GPU")
  else:
    dev = torch.device("cpu")
    print("Running on the CPU")
  return dev

def config_dimentions_and_dierctories():
  ### Get Time ####################################
  today = datetime.today()
  now = datetime.now()
  strToday = today.strftime("%m.%d.%y")
  strNow = now.strftime("%H:%M:%S")
  strTime = strToday + "_" + strNow
  print("Current Time =", strTime)
  #################################################

  ######### H in Latent Space #######
  ### H initial #####
  H_matrix_for_visual=H
  b_for_visual=b
  # define H
  h_lower_dimension = h_latent_space(H_matrix_for_visual, b_for_visual, encoded_dimention, m)
  ########################################

  ################# directories names ####################
  print("0. Load trained encoder")
  data_name = "Lorenz_visual"
  folder_KNet_name = './KNet/{}/'.format(data_name)
  if prior_flag:
      model_conv_trained = Encoder_conv_with_prior(m)
  else:
      model_conv_trained = Encoder_conv(m)
  if dataset_configuration ==1:
    data_configuration_name = 'lorenz_T=200_q={}_r={}.pt'.format(real_q,real_r)
    if prior_flag:
        sinerio = 'with_prior_baseline'
        folder_KGain_trained = folder_KNet_name + 'KGain_optimal_' + data_name + '_with_prior_baseline_fix_enc_' + str(0) + '_fix_h_' + str(0) + '_r_' + str(real_r) + '.pt'
        model_conv_trained.load_state_dict(torch.load('./saved_models/snr_range/200_length_with_prior/lorenz_Only_encoder_r={}_prior_r={}.pt'.format(real_r,prior_r)),strict=False)
    else:
        sinerio = 'baseline'
        folder_KGain_trained = folder_KNet_name + 'KGain_optimal_' + data_name +'_baseline_fix_enc_' + str(0)+'_fix_h_' + str(0)+'_r_'+str(real_r)+'.pt'
        model_conv_trained.load_state_dict(torch.load('./saved_models/snr_range/200_length/lorenz_Only_encoder_r={}.pt'.format(real_r)), strict=False)
  if dataset_configuration == 2:
    data_configuration_name = 'lorenz_T=200_decimated_q={}_r={}.pt'.format(real_q, real_r)
    if prior_flag:
        sinerio = 'with_prior_decimated'
        folder_KGain_trained = folder_KNet_name + 'KGain_optimal_' + data_name + 'with_prior_decimated_fix_enc_' + str(fix_encoder_flag) + '_fix_h_' + str(fix_H_flag) + '_r_' + str(real_r) + '.pt'
        model_conv_trained.load_state_dict(torch.load('./saved_models/snr_range/200_length_decimated_with_prior/lorenz_Only_encoder_decimated_r={}_prior_r={}.pt'.format(real_r,prior_r)),strict=False)
    else:
        sinerio = 'decimated'
        folder_KGain_trained = folder_KNet_name + 'KGain_optimal_' + data_name +'_decimated_fix_enc_' + str(fix_encoder_flag)+'_fix_h_' + str(fix_H_flag)+'_r_'+str(real_r)+'.pt'
        model_conv_trained.load_state_dict(torch.load('./saved_models/snr_range/200_length_decimated/lorenz_Only_encoder_decimated_r={}.pt'.format(real_r)),strict=False)
  if dataset_configuration == 3:
    data_configuration_name = 'lorenz_T=2000_q={}_r={}.pt'.format(real_q,real_r)
    if prior_flag:
        sinerio = 'with_prior_baseline'
        folder_KGain_trained = folder_KNet_name + 'KGain_optimal_' + data_name + '_with_prior_baseline_fix_enc_' + str(0) + '_fix_h_' + str(0) + '_r_' + str(real_r) + '.pt'
        model_conv_trained.load_state_dict(torch.load('./saved_models/snr_range/200_length_with_prior/lorenz_Only_encoder_r={}_prior_r={}.pt'.format(real_r,prior_r)), strict=False)
    else:
        sinerio = 'baseline'
        folder_KGain_trained = folder_KNet_name + 'KGain_optimal_' + data_name +'_baseline_fix_enc_' + str(0)+'_fix_h_' + str(0)+'_r_'+str(real_r)+'.pt'
        model_conv_trained.load_state_dict(torch.load('./saved_models/snr_range/200_length/lorenz_Only_encoder_r={}.pt'.format(real_r)),strict=False)

  folder_simulations = './Simulations/{}/'.format(data_name)
  folder_EKF_results = './EKF_results_on_test/{}/'.format(data_name)

  return strTime, sinerio ,h_lower_dimension, model_conv_trained, data_configuration_name, data_name, folder_KNet_name, folder_EKF_results, folder_simulations,folder_KGain_trained, H_matrix_for_visual

def print_weights(Knet_pipeline, state):
  print("encoder {} training".format(state))
  if fix_encoder_flag:
    print("fix encoder")
  else:
    print("trainable encoder")
  print(Knet_pipeline.model.model_encoder.encoder_conv[0].weight)

  print("H {} training".format(state))
  if fix_H_flag:
    print("fix H")
  else:
    print("trainable h in latent space")
  print(Knet_pipeline.model.h_learned_lower_dimension.fc.weight)
  print(Knet_pipeline.model.h_learned_lower_dimension.fc.bias)



