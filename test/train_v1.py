### test script for training
import time
import math
from datetime import datetime
import numpy as np
from new_env import CassieEnv
from ddpg_ac import Agent
from a_c_test_agent import NewAgent
from cassiemujoco import *

model = '/home/fury/OstrichCassie/Cassie_Ostrich_RL/test/cassie.xml'
traj_path = 'ostrichrl/ostrichrl/assets/mocap/cassie/'
bot = CassieSim(model,terrain = False)

env = CassieEnv(model,traj_path,60)


agent = Agent(alpha=0.0025, beta=0.0025, input_dims=[104], tau=0.001, env=env,
              batch_size=128,  layer1_size=512, layer2_size=512, n_actions=20)
#agent.load_models()
#agent.check_actor_params()
'''
try: 
        agent.load_models()
        agent.check_actor_params()
except:
        print("no saved models found!!")
'''
score_history = []
i = 0
tot_episodes = 1
max_tp = 60*60*10*10*2
tp = 0
for i in range(tot_episodes):
        obs = env.reset()
        done = False
        score = 0
        while tp != max_tp:
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
                act = agent.choose_action(obs)
                #print(env.step(act))
                new_state, reward, done, extra = env.step(act)
                #print(new_state[0:3])
                #print(env.sim.xpos('cassie-pelvis'))
                env.render()
                print(act.shape)
                agent.remember(obs, act, reward, new_state, int(done))
                agent.learn()
                score += reward
                obs = new_state
                time.sleep(0.2)
                #env.render()
                
                print('timestep: ', tp,'sim time: %.2f'% env.time,' reward: ',env.reward)
                tp = tp + 1
        score_history.append(score)
        print('episode: ', i,'score: %.2f' % score,'sim time: %.2f'% env.time,' reward: ',env.reward_t)
        #print('sim time: %.2f'% env.sim_time,' reward: ',env.reward_t)
        #print(env.obs_t, env.action)
        #print(env.reward_t)
print(agent.actor.checkpoint_file)
agent.save_models()      
#plotLearning(score_history, filename = 'plot1.png', window=20)
print("DONE lol")

