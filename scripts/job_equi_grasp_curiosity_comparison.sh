run()
{
jn=${env_abbr}_${date}_${folder}
script=run_short.sbatch

args="
--init_curiosity_l2=${init_curiosity_l2}
--final_curiosity_l2=${final_curiosity_l2}
--onlyfailure=8
--td_err_measurement=${td_err_measurement}
--explore=${explore}
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

date=0213
env_abbr=eqvar_grasp

for runs in 1 2
  do
      td_err_measurement=smooth_l1
      folder=01_
      explore=500
      init_curiosity_l2=0.1
      final_curiosity_l2=0.1
      run

      folder=1_0_100_
      explore=100
      init_curiosity_l2=1
      final_curiosity_l2=0.
      run

      folder=1_0_200_
      explore=200
      init_curiosity_l2=1
      final_curiosity_l2=0.
      run

      folder=1_0_400_
      explore=400
      init_curiosity_l2=1
      final_curiosity_l2=0.
      run

      td_err_measurement=BCE
      folder=_RSS_BCE_
      explore=500
      init_curiosity_l2=0
      final_curiosity_l2=0
      run

      folder=01_BCE_
      explore=500
      init_curiosity_l2=0.1
      final_curiosity_l2=0.1
      run

      folder=1_0_100_BCE_
      explore=100
      init_curiosity_l2=1
      final_curiosity_l2=0.
      run

      folder=1_0_200_BCE_
      explore=200
      init_curiosity_l2=1
      final_curiosity_l2=0.
      run

      folder=1_0_400_BCE_
      explore=400
      init_curiosity_l2=1
      final_curiosity_l2=0.
      run

  done