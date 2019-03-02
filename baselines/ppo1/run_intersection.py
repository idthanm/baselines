#!/usr/bin/env python3
import os
from baselines.common.cmd_util import common_arg_parser
from baselines.common import tf_util as U
from baselines import logger
from baselines.ppo1.Environment import environment
import gym


def train(env, num_timesteps, load_model_path=None):
    from baselines.ppo1 import mlp_policy, pposgd_simple
    U.make_session(num_cpu=1).__enter__()

    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                    hid_size=128, num_hid_layers=2)


    # parameters below were the best found in a simple random search
    # these are good enough to make humanoid walk, but whether those are
    # an absolute best or not is not certain
    # env = RewScale(env, 0.1)
    pi = pposgd_simple.learn(env, policy_fn,
                             max_timesteps=num_timesteps,
                             timesteps_per_actorbatch=2048,
                             clip_param=0.2, entcoeff=0.0,
                             optim_epochs=10,
                             optim_stepsize=3e-4,
                             optim_batchsize=64,
                             gamma=0.99,
                             lam=0.95,
                             schedule='linear',
                             load_model_path=load_model_path
                             )

    return pi

# class RewScale(gym.RewardWrapper):
#     def __init__(self, env, scale):
#         gym.RewardWrapper.__init__(self, env)
#         self.scale = scale
#     def reward(self, r):
#         return r * self.scale

def main():
    logger.configure('E:\\Project\\Toyota RL\\Toyata 2018\\Toyata RL 4th quarter\\log')
    # 'F:\\GuanYang\\toyota2018_4\\log'
    parser = common_arg_parser()
    parser.add_argument('--load_model_path', default=None)
    parser.set_defaults(num_timesteps=int(2e7))

    args = parser.parse_args()
    env = environment.Env(N=6, pattern=[0, 2, 4, 8, 9, 10], height=30, width=30)

    if not args.play:
        # train the model
        train(env=env, num_timesteps=args.num_timesteps, load_model_path=args.load_model_path)
    else:
        # construct the model object, load pre-trained model and render
        pi = train(env=env, num_timesteps=1)
        U.load_state(args.load_model_path)
        ob = env.manualSet(modelList=env.pattern)
        while True:
            action = pi.act(stochastic=False, ob=ob)[0]
            # ob, _, done, _ =  env.step(action)
            ob, rew, done, _ = env.updateEnv(action)
            env.showEnv()
            if done:
                ob = env.manualSet(modelList=env.pattern)


if __name__ == '__main__':
    main()
