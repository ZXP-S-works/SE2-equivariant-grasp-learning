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
--dilation_aperture=${dilation_aperture}
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

date=0103
env_abbr=dilation_aperture

for dilation_aperture in 4 6 8 10
  do
      alg=dqn_asr
      model=equ_resu_nodf_flip_softmax
      q2_model=equ_shift_reg_7_lq_softmax_resnet64
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

  done