activate tensorflow
mpiexec -np 8 python -m baselines.ppo1.run_intersection --model-path=./toyota/model/intersection_policy --num_timesteps=int(1e6)