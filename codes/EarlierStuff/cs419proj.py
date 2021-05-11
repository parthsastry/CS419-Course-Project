import gym
import matplotlib.pyplot as plt
import time
import matplotlib 

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython: from IPython import display


num_steps = 1500

env = gym.make('Breakout-v0')
observation_space = env.observation_space
action_space = env.action_space
print("The observation space: {}".format(observation_space))
print("The action space: {}".format(action_space))

observation = env.reset()
print('The initial observation is {}'.format(observation))

for step in range(num_steps):

	action = env.action_space.sample()
	new_observation, reward, done, info = env.step(action)
	a = env.render()
	plt.figure()
	plt.imshow(a)
	plt.show()
	if is_ipython: display.clear_output(wait=True)

	time.sleep(0.01)

	if done:
		env.reset()