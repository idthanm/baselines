from baselines.ppo1.Environment import environment
from baselines.ppo1 import run_intersection
if __name__ == '__main__':
    env = environment.Env(3, 0.1, 30)
    print(env.action_space.sample(), env.observation_space.sample())
    print(run_intersection.__name__)
    print('E:\Research\Reinforcement Learning\openai_baseline\\baselines\\toyota\summary')