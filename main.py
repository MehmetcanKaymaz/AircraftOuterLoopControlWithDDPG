import sys
import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ddpg import DDPGagent
from utils import *
from Environment import Env
import datetime

env = Env()

agent = DDPGagent(env)
noise = OUNoise(env.action_space,env.action_space_min,env.action_space_max)
batch_size = 1000
rewards = []
avg_rewards = []

for episode in range(100000):
    state = env.reset()
    state=np.array(state)
    noise.reset()
    episode_reward = 0

    for step in range(1000):
        action = agent.get_action(state)
        action = noise.get_action(action, step)
        new_state, reward, done = env.step(action)
        new_state=np.array(new_state)
        agent.memory.push(state, action, reward, new_state, done)

        if len(agent.memory) > batch_size:
            agent.update(batch_size)

        state = new_state
        episode_reward += reward

        if done:
            sys.stdout.write(
                "episode: {}, reward: {}, average _reward: {} , Distance={}\n".format(episode, np.round(episode_reward, decimals=2),
                                                                         np.mean(rewards[-10:]) ,env.distance))
            break

    rewards.append(episode_reward)
    avg_rewards.append(np.mean(rewards[-100:]))
    if episode%1000==0:
        agent.save_models()


agent.save_models()
np.savetxt("Outputs/reward"+str(datetime.datetime.now())+".txt",rewards)
np.savetxt("Outputs/avg_reward"+str(datetime.datetime.now())+".txt",avg_rewards)

"""
plt.plot(rewards)
plt.plot(avg_rewards)
plt.plot()
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()"""
