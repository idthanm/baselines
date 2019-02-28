activate tensorflow
cd E:\Research\Reinforcement Learning\openai_baseline\baselines
mpiexec -np 8 python -m baselines.ppo1.run_intersection --load_model_path="E:\Project\Toyota RL\Toyata 2018\Toyata RL 4th quarter\model\intersection_policy-" --num_timesteps=1e6
cd C:\Users\GuanYang\PycharmProjects\toyota2018_4
mpiexec -np 16 python -m baselines.ppo1.run_intersection --load_model_path=F:\GuanYang\toyota2018_4\model\intersection_policy- --num_timesteps=2e7
