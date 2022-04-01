run()
{
jn=${env_abbr}_${date}_${folder}
script=run_short.sbatch
#script=run_short_normal_gpu.sbatch
#script=run_local_test.sbatch

args="
--action_selection=${action_selection}
--alg=${alg}
--model=${model}
--equi_n=${equi_n}
--q1_failure_td_target=${q1_failure_td_target}
--init_eps=${init_eps}
--final_eps=${final_eps}
--max_episode=1500
--step_eps=${step_eps}
--batch_size=${batch_size}
--training_offset=${training_offset}
--q2_model=${q2_model}
--q2_train_q1=${q2_train_q1}
--q2_input=${q2_input}
--num_rotations=${num_rotations}
--aug=${aug}
--onpolicy_data_aug_n=${onpolicy_data_aug_n}
--training_iters=${training_iters}
--sample_onpolicydata=${sample_onpolicydata}
--onlyfailure=${onlyfailure}
"
slurm_args=""

#--load_buffer=/scratch/zhu.xup/project/eqvar_grasp/dataset/${buffer}.pt

jn1=${jn}_1
jid[1]=$(sbatch ${slurm_args} --job-name=${jn1} --export=LOAD_SUB='None' ${script} ${args} | tr -dc '0-9')
#source ${script} ${args}
#for j in {2..3}
#do
#jid[${j}]=$(sbatch ${slurm_args} --job-name=${jn}_${j} --dependency=afterok:${jid[$((j-1))]} --export=LOAD_SUB=${jn}_$((j-1))_${jid[$((j-1))]} ${script} ${args} | tr -dc '0-9')
#done
}

date=0114
env_abbr=eqvar_grasp

for runs in 1 2
  do
      folder=no_equ
      alg=dqn_asr
      model=resu_like_equ_resu
      equi_n=4
      q2_model=cnn_like_equ_lq_softmax_resnet64
      q1_failure_td_target=non_action_max_q2
      q2_input=hm_minus_z
      q2_train_q1=Boltzmann10
      action_selection=Boltzmann
      init_eps=1.
      final_eps=0.
      step_eps=20
      training_offset=20
      aug=0
      onpolicy_data_aug_n=8
      num_rotations=8
      batch_size=8
      sample_onpolicydata=True
      onlyfailure=4
      training_iters=1
#      run

      folder=no_asr
      alg=dqn_fcn_si
      model=equ_resu_nodf_softmax
      q2_model=None
      equi_n=16
      q1_failure_td_target=non_action_max_q2
      q2_input=hm_minus_z
      q2_train_q1=Boltzmann10
      action_selection=Boltzmann
      init_eps=1.
      final_eps=0.
      step_eps=20
      training_offset=20
      aug=0
      onpolicy_data_aug_n=8
      num_rotations=8
      batch_size=8
      sample_onpolicydata=True
      onlyfailure=4
      training_iters=1
#      run

      folder=no_opt
      alg=dqn_asr
      model=equ_resu_nodf_flip
      equi_n=4
      q2_model=equ_shift_reg_7_lq_resnet64
      q1_failure_td_target=rewards
      q2_input=hm
      q2_train_q1=None
      action_selection=egreedy
      init_eps=0.5
      final_eps=0.1
      step_eps=0
      training_offset=0
      aug=0
      onpolicy_data_aug_n=0
      num_rotations=8
      batch_size=8
      sample_onpolicydata=f
      onlyfailure=0
      training_iters=1
#      run

      folder=Rot_FCN
      alg=dqn_fcn
      model=resu_like_equ_resu
      q2_model=None
      q1_failure_td_target=non_action_max_q2
      q2_input=hm_minus_z
      q2_train_q1=Boltzmann10
      action_selection=Boltzmann
      init_eps=1.
      final_eps=0.
      step_eps=20
      training_offset=20
      aug=0
      onpolicy_data_aug_n=8
      num_rotations=8
      batch_size=8
      sample_onpolicydata=True
      onlyfailure=4
      training_iters=1
      run

      folder=2RAD_Rot_FCN
      alg=dqn_fcn
      model=resu_like_equ_resu
      q2_model=None
      q1_failure_td_target=non_action_max_q2
      q2_input=hm_minus_z
      q2_train_q1=Boltzmann10
      action_selection=Boltzmann
      init_eps=1.
      final_eps=0.
      step_eps=20
      training_offset=20
      aug=1
      onpolicy_data_aug_n=8
      num_rotations=8
      batch_size=8
      sample_onpolicydata=True
      onlyfailure=4
      training_iters=2
      run

      folder=4RAD_Rot_FCN
      alg=dqn_fcn
      model=resu_like_equ_resu
      q2_model=None
      q1_failure_td_target=non_action_max_q2
      q2_input=hm_minus_z
      q2_train_q1=Boltzmann10
      action_selection=Boltzmann
      init_eps=1.
      final_eps=0.
      step_eps=20
      training_offset=20
      aug=1
      onpolicy_data_aug_n=8
      num_rotations=8
      batch_size=8
      sample_onpolicydata=True
      onlyfailure=4
      training_iters=4
      run


      folder=8RAD_Rot_FCN
      alg=dqn_fcn
      model=resu_like_equ_resu
      q2_model=None
      q1_failure_td_target=non_action_max_q2
      q2_input=hm_minus_z
      q2_train_q1=Boltzmann10
      action_selection=Boltzmann
      init_eps=1.
      final_eps=0.
      step_eps=20
      training_offset=20
      aug=1
      onpolicy_data_aug_n=8
      num_rotations=8
      batch_size=8
      sample_onpolicydata=True
      onlyfailure=4
      training_iters=8
      run

  done