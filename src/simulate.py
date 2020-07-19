import gym
from model import Agent
import torch as T
import time

model = T.load('model/model.pt', map_location=T.device('cpu'))
env = gym.make('LunarLander-v2')
# env.monitor.start('cartpole-experiment-1', force=True)
agent = Agent(gamma=0.99, epsilon=0.01, batch_size=64, n_actions=4, eps_end=0.01,
                  input_dims=[8], lr=0.001)

agent.Q_eval.load_state_dict(model)

for i in range(13):
    done = False
    observation = env.reset()
    k = 0
    tot = 0
    while not done:
        time.sleep(.005)
        action = agent.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        observation = observation_
        env.render()
        k += 1
        tot += reward 
    print('episode:', i, '; steps:', k, '; tot reward:', tot, "; sub:", tot-k)

env.close()