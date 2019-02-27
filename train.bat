activate tensorflow
mpiexec -np 8 python -m baselines.ppo1.run_intersection --load_model_path=./toyota/model/intersection_policy-35864 --num_timesteps=1e6

mpiexec -np 16 python -m baselines.ppo1.run_intersection --num_timesteps=2e7
