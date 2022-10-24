# LATENT_ICASSP23

## How to run the code
python3 main_visual.py

### flags:
##### data
* 'dataset_configuration': 1 - Ttrain=Ttest=200, 2 - Ttrain=200 Ttest=200 but decimated, 3 - Ttrain=200 Ttest=2000
* 'data_gen_flag': True - Generating Dataset. False - Loading Dataset
##### Knet
* 'load_KGain_trained_flag': True - loading trained KGain model. False - KGain model start from scratch
* 'flag_Train': True - Training full pipeline. False - Only inference
##### EKF
* 'evaluate_EKF_flag': True - Evaluate EKF results. False - Load EKF results

### configurations:
Under /configurations/config_file.yaml, you can change 
##### Hyper Parameters
* lr_kalman : 1e-5
* wd_kalman : 1e-2
* batch_size : 32
* epoches : 400
##### Noise statistics
* real_r : 4
* real_q : 0.1
