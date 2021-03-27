import sys
import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ddpg import DDPGagent
from utils import *
from Environment import Env

env = Env()

agent = DDPGagent(env)
noise = OUNoise(env.action_space,env.action_space_min,env.action_space_max)
batch_size = 128
rewards = []
avg_rewards = []

for episode in range(10):
    state = env.reset()
    state=np.array(state)
    noise.reset()
    episode_reward = 0

    for step in range(1000):
        action = agent.get_action(state)
        action = noise.get_action(action, step)
        #print(action)
        new_state, reward, done = env.step(action)
        new_state=np.array(new_state)
        agent.memory.push(state, action, reward, new_state, done)

        if len(agent.memory) > batch_size:
            #print("Learning Started")
            agent.update(batch_size)

        state = new_state
        episode_reward += reward

        if done:
            sys.stdout.write(
                "episode: {}, reward: {}, average _reward: {} , Distance={}\n".format(episode, np.round(episode_reward, decimals=2),
                                                                         np.mean(rewards[-10:]) ,env.distance))
            break

    rewards.append(episode_reward)
    avg_rewards.append(np.mean(rewards[-10:]))


agent.save_models()
"""
plt.plot(rewards)
plt.plot(avg_rewards)
plt.plot()
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()"""