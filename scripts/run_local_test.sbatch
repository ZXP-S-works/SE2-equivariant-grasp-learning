#!/bin/bash

export PYTHONPATH="$PYTHONPATH:/home/zxp-s-works/helping_hands_rl_envs"
source ~/.bashrc
# conda activate eqvar_grasp

args="
--env=house_building_1
--simulator=pybullet
--robot=kuka
--alg=dqn
--model=resucat
--gamma=0.9
--perlin=0.02
--init_eps=0.5
--planner_episode=10000
--explore=10000
--num_objects=3
--no_bar
--num_process=1
--max_episode=50000
--batch_size=32
"

m_run=1

for (( i=1; i<=$m_run; i++ ))
do
    log=${SLURM_JOB_NAME}_${SLURM_JOB_ID}_${i}
    if [ "$LOAD_SUB" = "None" ]; then
    	load="None"
    else
    	load=${LOAD_SUB}_${i}
    fi

    python main.py --log_sub=${log} --load_sub=${load} ${args} $@ &
    PIDS+=($!)
done

# wait for all processes to finish, and store each process's exit code into array STATUS[].
for pid in ${PIDS[@]}; do
  echo "pid=${pid}"
  wait ${pid}
  STATUS+=($?)
done

# after all processed finish, check their exit codes in STATUS[].
i=0
for st in ${STATUS[@]}; do
  if [[ ${st} -ne 0 ]]; then
    echo "$i failed"
    exit 1
  else
    echo "$i finish"
  fi
  ((i+=1))
done

exit
