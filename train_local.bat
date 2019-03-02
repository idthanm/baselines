::before you play or train, please make sure you save the log files!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
call activate tensorflow
e:
cd E:\Research\Reinforcement Learning\openai_baseline\baselines
mpiexec -np 8 python -m baselines.ppo1.run_intersection --num_timesteps=2e8
::mpiexec -np 8 python -m baselines.ppo1.run_intersection --load_model_path="E:\Project\Toyota RL\Toyata 2018\Toyata RL 4th quarter\model\intersection_policy-" --num_timesteps=2e8
