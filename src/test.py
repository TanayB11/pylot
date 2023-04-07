import gym
import gym_jsbsim

env = gym.make("")
env.reset()
done = False

while not done:
   action = env.action_space.sample()
   state, reward, done, _ = env.step(action)
