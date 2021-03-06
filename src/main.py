import gym
from model import Agent
from utils import plotLearning
import numpy as np
import torch as T

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=256, n_actions=4, eps_end=0.01,
                  input_dims=[8], lr=0.001)
    scores, eps_history = [], []
    n_games = 12000
    
    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()
        steps = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward-(steps/100), 
                                    observation_, done)
            agent.learn()
            observation = observation_
            steps += 1
        scores.append(score)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(scores[-100:])
        print('episode ', i, 'score %.2f' % score,
                'average score %.2f' % avg_score,
                'epsilon %.2f' % agent.epsilon)

    T.save(agent.Q_eval.state_dict(), 'model/model.pt')

