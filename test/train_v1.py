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

#model = '/home/k38/Cassie_Ostrich_RL/test/cassie.xml'
model = '/home/fury/LIDAR/Cassie_Ostrich_RL/test/cassie.xml'
traj_path = 'ostrichrl/ostrichrl/assets/mocap/cassie/'
bot = CassieSim(model,terrain = False)

env = CassieEnv(model,traj_path,simrate=120)



agent = Agent(alpha=0.00001, beta=0.00001, input_dims=[56], tau=0.01, env=env,
              batch_size=128,  layer1_size=1024, layer2_size=1024, n_actions=10)
agent.load_models()
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
tot_episodes = 200000
max_tp = 60*60*10*10*2
tp = 0
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
                act = agent.choose_action(obs)
                #print(env.step(act))
                new_state, reward, done, extra = env.step(act)
                #print(new_state[0:3])
                #print(env.sim.xpos('cassie-pelvis'))
                env.render()
                print(new_state.shape)
                agent.remember(obs, act, reward, new_state, int(done))
                agent.learn()
                score += reward
                obs = new_state
                #time.sleep(1)
                #env.render()
                
                #print('timestep: ', tp,'sim time: %.2f'% env.time,' reward: ',env.reward)
                tp = tp + 1
        agent.learn()
        score_history.append(score)
        print('episode: ', i,'| score: %.2f' % score,' | reward: ',reward)

        if i%5000 == 0:
                print(agent.actor.checkpoint_file)
                agent.save_models()
        #print('sim time: %.2f'% env.sim_time,' reward: ',env.reward_t)
        #print(env.obs_t, env.action)
        #print(env.reward_t)

print(agent.actor.checkpoint_file)
agent.save_models()      
#plotLearning(score_history, filename = 'plot1.png', window=20)
print("DONE lol")

