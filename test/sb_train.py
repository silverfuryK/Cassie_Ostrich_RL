### test script for training
import time
import math
from datetime import datetime
import numpy as np
from new_env import CassieEnv
#from new_env import CassieEnv
from ddpg_ac import Agent
from a_c_test_agent import NewAgent
from cassiemujoco import *
import torch as th
print(th.__version__)
#rom stable_baselines3.common.policies import MlpPolicy
#from stable_baselines3.common import make_vec_env
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback


#model = '/home/k38/Cassie_Ostrich_RL/test/cassie.xml'
model = '/home/fury/LIDAR/Cassie_Ostrich_RL/test/cassie.xml'
traj_path = 'ostrichrl/ostrichrl/assets/mocap/cassie/'
bot = CassieSim(model,terrain = False)

env = CassieEnv(model,traj_path,simrate=120)
#check_env(env)



# agent = Agent(alpha=0.00001, beta=0.00001, input_dims=[56], tau=0.01, env=env,
#               batch_size=128,  layer1_size=1024, layer2_size=1024, n_actions=10)
# agent.load_models()
#agent.check_actor_params()
'''
try: 
        agent.load_models()
        agent.check_actor_params()
except:
        print("no saved models found!!")


'''

## Tensorboard logs ##

class TensorboardCallback(BaseCallback):
        """
        Custom callback for plotting additional values in tensorboard.
        """
        def __init__(self, verbose=0):
            super(TensorboardCallback, self).__init__(verbose)

        def _on_step(self) -> bool:                
            self.logger.record('reward/rew', np.mean(self.training_env.get_attr('reward')))
            self.logger.record('sim/time_steps', np.mean(self.training_env.get_attr('time')))
        #     self.logger.record('reward/spring', np.mean(self.training_env.get_attr('rew_spring_buf')))
        #     self.logger.record('reward/orientation', np.mean(self.training_env.get_attr('rew_ori_buf')))
        #     self.logger.record('reward/velocity', np.mean(self.training_env.get_attr('rew_vel_buf')))
        #     self.logger.record('reward/steps', np.mean(self.training_env.get_attr('time_buf')))
        #     self.logger.record('reward/totalreward', np.mean(self.training_env.get_attr('reward_buf')))  
        #     self.logger.record('reward/acc', np.mean(self.training_env.get_attr('rew_acc_buf')))    
        #     # self.logger.record('reward/ee', np.mean(self.training_env.get_attr('rew_ee_buf')))    
        #     self.logger.record('reward/action', np.mean(self.training_env.get_attr('rew_action_buf')))          
        #     self.logger.record('reward/perf', np.mean(self.training_env.get_attr('rew_ori_buf'))
        #                                     + np.mean(self.training_env.get_attr('rew_vel_buf')))
            
            if self.n_calls % 51200 == 0:
                print("Saving model")
                self.model.save(f"./models/PPO/ppo_cassie")

            return True

policy_kwargs = dict(activation_fn=th.nn.Tanh,
                     net_arch=dict(pi=[256,256,256], vf=[256, 256, 256]))

model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, tensorboard_log="./models/PPO/PPO_logs/", verbose=1)
#model.learn(total_timesteps=2e7)
model.save("models/PPO/ppo2_1")

model.load("models/PPO/ppo2_1")

model.learn(total_timesteps=6e7, callback=TensorboardCallback())

sobs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    #env.render()

