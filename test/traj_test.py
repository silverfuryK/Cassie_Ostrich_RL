### test script for training
import time
import math
from datetime import datetime
import numpy as np
from env_traj_test import CassieEnv
#from new_env import CassieEnv
from ddpg_ac import Agent
from a_c_test_agent import NewAgent
from cassiemujoco import *

#model = '/home/k38/Cassie_Ostrich_RL/test/cassie.xml'
model = '/home/fury/OstrichCassie/Cassie_Ostrich_RL/test/cassie.xml'
traj_path = 'ostrichrl/ostrichrl/assets/mocap/cassie/'
bot = CassieSim(model,terrain = False)

env = CassieEnv(model,traj_path,simrate=120)

tp = 0

tot_episodes = 200
for i in range(tot_episodes):
        obs = env.reset()
        #print(obs.shape)
        done = False
        score = 0
        #print('EPISODE ' + str(i))
        while not done:
                '''
                cmd_vel = trajec.get_cmd_vel(sim_time)
                #print(cmd_vel)
                #env.action([0,0,0.1,0,0,0.1])
                action = np.array(agent.choose_action(observation)).reshape((1,))
                print(action)
                #p.stepSimulation()
                observation_, reward, done = env.step_simulation(action)
                agent.learn(observation, reward, observation_, done)
                observation = observation_
                #print(done)
                #print([env.obs_t[2],env.obs_t[3],env.obs_t[6],env.obs_t[7],env.obs_t[11]])
                score += reward
                #time.sleep(dt)
                '''
                
                #print(env.step(act))
                new_state, reward, done, extra = env.step()
                #print(new_state[0:3])
                #print(env.sim.xpos('cassie-pelvis'))
                env.render()
                #print(act.shape)
                #time.sleep(1)
                #env.render()
                
                #print('timestep: ', tp,'sim time: %.2f'% env.time,' reward: ',env.reward)
                tp = tp + 1
