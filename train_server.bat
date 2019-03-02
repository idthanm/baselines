::before you play or train, please make sure you save the log files!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

call activate tensorflow
c:
cd C:\Users\GuanYang\PycharmProjects\toyota2018_4
mpiexec -np 16 python -m baselines.ppo1.run_intersection --num_timesteps=2e8
::mpiexec -np 16 python -m baselines.ppo1.run_intersection --load_model_path="F:\GuanYang\toyota2018_4\model\intersection_policy-" --num_timesteps=2e8
