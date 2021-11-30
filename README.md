# Equivariant Grasp Learning In Real Time

**Abstract**

Visual grasp detection is a key problem in robotics where the agent must learn to model the grasp function, a mapping 
from an image of a scene onto a set of feasible grasp poses. In this paper, we recognize that the SE(2) grasp function 
is SE(2)-equivariant and that it can be modeled using an equivariant convolutional neural network. As a result, we are
able to significantly improve the sample efficiency of grasp learning to the point where we can learn a good
approximation of the SE(2) grasp function within only 500 grasp experiences. This is fast enough that we can learn to
grasp completely on a physical robot in about an hour. 


**Citation**





## Environments

### Simulation Environment

The simulation environment is random_household_picking_clutter_full_obs_30. This environment is implemented in asrse3/helping_hands_rl_envs/envs/pybullet_envs.

![alt text](./images/simulation_env.png)

### Physical Environment
The physical robot environment is DualBinFrontRear. To train on this environment, a physical robot set up is required.

![alt text](./images/UR5_setup.png)


## Installation

**Step 1**. Install Miniconda with Python==3.7 (If you have installed anaconda/miniconda previously, please jump to **Step 2.**)

```bash
# Tested on Ubuntu 20.04, x86_64 platform. Other linux distributions should work. Other platform users please refer to "https://docs.conda.io/en/latest/miniconda.html" for platform-specific miniconda downloading
sudo apt update
sudo apt install curl
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# enter "yes" if any settings appears
# Reopen your Terminal
# Run the command "conda list". A list of installed packages appears if it has been installed correctly.
```

**Step 2**. Create a Python3.7 environment with Miniconda and Pytorch installation

```bash
# If your miniconda is successfully installed, you will see "(base)" in front of your username
conda create -n rtgrasp python==3.7 -y
conda activate rtgrasp
git clone https://github.com/ZXP-S-works/asrse3.git
cd ./asrse/asrse3

# CUDA 11.0
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

# CUDA 10.2
pip install torch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2

# CUDA 10.1
pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

# CUDA 9.2
pip install torch==1.7.1+cu92 torchvision==0.8.2+cu92 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

# CPU only
pip install torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html


```

**Step 3**. Install dependencies

```
# Tested on python==3.7, pytorch==1.7.1, torchvision==0.8.2, pybullet==2.7.1, as you have installed as you follow instructions above
pip install -r requirements.txt
```



## Reinforcement learning


### Training baselines in simulation

Our method
```bash
python3 ./scripts/main.py 
```

Random policy (with heuristic)
```bash
python3 ./scripts/main.py --model=resu --q2_model=cnn_no_sharing --explore=10000 --training_offset=10000 --init_eps=1. --final_eps=1. 
```

FCN (4RAD FCN)
```bash
python3 ./scripts/main.py --alg=dqn_fcn --model=resu_softmax --q2_model=cnn_no_sharing_softmax --aug=1 --training_iters=4 --action_selection=egreedy --init_eps=0.5 --final_eps=0.1 --q1_failure_td_target=rewards --sample_onpolicydata=f --onlyfailure=f
```

Rot FCN (4RAD Rot FCN)
```bash
python3 ./scripts/main.py --alg=dqn_fcn_si --model=resu_softmax --q2_model=cnn_no_sharing_softmax --aug=1 --training_iters=4 --action_selection=egreedy --init_eps=0.5 --final_eps=0.1 --q1_failure_td_target=rewards --sample_onpolicydata=f --onlyfailure=f
```

ASR (8RAD ASR)
```bash
python3 ./scripts/main.py --alg=dqn_asr --model=resu_softmax --q2_model=cnn_no_sharing_softmax --aug=1 --training_iters=8 --action_selection=egreedy --init_eps=0.5 --final_eps=0.1 --q1_failure_td_target=rewards --sample_onpolicydata=f --onlyfailure=f
```

Default parameters

```
--env=random_household_picking_clutter_full_obs_30
--num_processes=1
--eval_num_processes=10
--render=f # set it to True to see the actual simulation & training process
--learning_curve_avg_window=50
--training_offset=20
--target_update_freq=20
--q1_failure_td_target=non_action_max_q2
--q1_success_td_target=rewards
--alg=dqn_asr
--model=equ_resu_nodf_flip_softmax
--q2_train_q1=Boltzmann10
--q2_model=equ_shift_reg_7_lq_softmax_last_no_maxpool32
--q2_input=hm_minus_z
--q3_input=hm_minus_z
--patch_size=32 
--batch_size=8
--max_episode=1500
--explore=500
--action_selection=Boltzmann
--hm_threshold=0.005
--step_eps=0
--init_eps=0.
--final_eps=0. 
--log_pre=../results/household_repo/rand_household_picking_clutter/ 
--sample_onpolicydata=t 
--onlyfailure=t 
--num_rotations=8 
--aug=0
--onpolicy_data_aug_n=8 
--onpolicy_data_aug_flip=True 
--onpolicy_data_aug_rotate=True 
--num_eval_episodes=1000
```

### Real-time training in a physical robot
The parallel training is only implemented in physical robot environment. However, one can easily modify it to any environment.

```bash
python3 ./scripts/train_robot_parallel.py --env=DualBinFrontRear --num_processes=1 --eval_num_processes=10 --render=f --learning_curve_avg_window=50 --training_offset=20 --target_update_freq=20 --q1_failure_td_target=non_action_max_q2 --q1_success_td_target=rewards --alg=dqn_asr --model=equ_resu_nodf_flip_softmax --q2_train_q1=Boltzmann10 --q2_model=equ_shift_reg_7_lq_softmax_last_no_maxpool32 --q2_input=hm_minus_z --q3_input=hm_minus_z --patch_size=32 --batch_size=16 --max_episode=1000 --explore=500 --action_selection=Boltzmann --hm_threshold=0.015 --step_eps=0 --init_eps=0. --final_eps=0. --log_pre=../results/household_repo/rand_household_picking_clutter/ --sample_onpolicydata=t --onlyfailure=t --num_rotations=8 --aug=f --onpolicy_data_aug_n=16 --onpolicy_data_aug_flip=True --onpolicy_data_aug_rotate=True
```

The figure illustrating the parallel training is as follows:

![alt text](./images/dualbin_paralle_run.png)


## Supervised learning

Our method
```bash
python3 ./supervised_learning/train_network.py --dataset cornell --heightmap_size=96 --action_pixel_range=96 --network=ours_method --train_with_centers=f --normalize_depth=True --batches-per-epoch=100 --use_length=400 --use-depth=1 --use-rgb=0 --num-workers=0 --split=0.75 --dataset-path=/home/zxp-s-works/robotic-grasping/dataset/Cornell_Grasping_Dataset --description training_cornell --alg=dqn_asr --model=equ_resu_nodf_flip_softmax --q1_train_q2=2 --q2_model=equ_shift_reg_7_lq_softmax_last_no_maxpool32 --q2_predict_width=t --patch_size=32 --log_pre=../results/household_repo/rand_household_picking_clutter/sl/ --num_rotations=16
```

GR-ConvNet
```bash
python3 ./supervised_learning/train_network.py --dataset cornell --network=grconvnet3 --normalize_depth=True --batches-per-epoch=100 --use_length=400 --use-depth=1 --use-rgb=0 --num-workers=0 --split=0.75 --dataset-path=/home/zxp-s-works/robotic-grasping/dataset/Cornell_Grasping_Dataset --description training_cornell --log_pre=../results/household_repo/rand_household_picking_clutter/sl/
```
