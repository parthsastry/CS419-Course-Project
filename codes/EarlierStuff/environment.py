import gym
import numpy as np
import time
import argparse

class Environment(object):
    def __init__(self, env = 'Breakout-v0', test=False):
        self.env = gym.make(env)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def seed(self, seed):
        self.env,seed(seed)

    def reset(self):
        return np.array(self.env.reset())

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return np.array(observation), reward, done, info

    def get_action_space(self):
        return self.action_space

    def get_observation_space(self):
        return self.observation_space

    def get_random_action(self):
        return self.action_space.sample()

    def render(self):
        self.env.render()

def testEnvironmentSetup():
    en = Environment()
    en.reset()
    for step in range(1000):

        action = en.get_random_action()
        new_observation, reward, done, info = en.step(action)
        en.render()
        time.sleep(0.01)

        if done:
            en.reset()

###################################################################

testEnvironmentSetup()
