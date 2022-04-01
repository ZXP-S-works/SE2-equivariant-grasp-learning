run()
{
jn=${env_abbr}_${date}_${description}
script=run_sl8.sbatch
args="
--dataset=jacquard
--description=${description}
--use-dropout=0
--env=none
--alg=${alg}
--model=${model}
--model_predict_width=t
--batch_size=${batch_size}
--q1_train_q2=8
--q2_model=${q2_model}
--q2_predict_width=t
--q2_input=hm
--patch_size=32
--num_rotations=16
--train_tau=0.1
--heightmap_size=96
--action_pixel_range=96
--max_episode=50
--num_eval=50
--batches-per-epoch=${batches_per_epoch}
--train_with_y_pos=f
--normalize_depth=${normalize_depth}
--train_size=${train_size}
--test_size=256
--use-depth=1
--use-rgb=1
--num-workers=4
--dataset-path=../dataset/Jacquard
--log_pre=/scratch/zhu.xup/project/eqvar_grasp/results/sl/
"
slurm_args=""

jn1=${jn}_1
jid[1]=$(sbatch ${slurm_args} --job-name=${jn1} --export=LOAD_SUB='None' ${script} ${args} | tr -dc '0-9')
#source ${script} ${args}
for j in {2..5}
do
jid[${j}]=$(sbatch ${slurm_args} --job-name=${jn}_${j} --dependency=afterok:${jid[$((j-1))]} --export=LOAD_SUB=${jn}_$((j-1))_${jid[$((j-1))]} ${script} ${args} | tr -dc '0-9')
done
}

date=0120
env_abbr=J_SL
#for train_size in 20 40 80 160 320 640
#for train_size in 8 32 128 512
#for train_size in 8 64 512 4096
#for train_size in 10 40 160 640
for train_size in 1024 256 64 16
  do
    batch_size=8
    batches_per_epoch=1000
    normalize_depth=t

    description=ours_RGBD
    alg=dqn_asr
    model=equ_resu_nodf_flip_softmax
    q2_model=equ_shift_reg_7_lq_softmax_resnet64
    run

    alg=dqn_fcn_si
    q2_model=None
    normalize_depth=f

    description=grconvnet_RGBD
    model=grconvnet
    run

    description=ggcnn_RGBD
    model=ggcnn
    run

    normalize_depth=t
    description=fcgqcnn_RGBD
    model=fcgqcnn
    run

    batch_size=1
    batches_per_epoch=100
    alg=dqn_fcn

    description=vpg_RGBD
    model=vpg
    run
  done