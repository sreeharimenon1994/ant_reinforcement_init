import gym
from model import Agent
from reward import RewardAgent
from utils import plotLearning
import numpy as np
import torch as T

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=32, n_actions=4,\
                  eps_end=0.01, input_dims=[8], lr=0.001)

    reward_model = RewardAgent(gamma=0.99, epsilon=1.0, batch_size=32,\
                            n_actions=4, eps_end=0.01, input_dims=[9], lr=0.001)

    scores, eps_history = [], []
    n_games = 12000
    
    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()
        pre_action = agent.choose_action(observation)
        steps = 0
        while not done:
            # env.render()
            tmp1 = observation
            tmp1 = np.append(tmp1, pre_action)
            r = reward_model.choose_action(tmp1)
            observation_, reward, done, info = env.step(pre_action)
            action = agent.choose_action(observation_)

            agent.store_transition(observation, action, r, 
                                    observation_, done)
            tmp2 = observation_
            tmp2 = np.append(tmp2, action)
            
            if done and reward < 0:
                print('predicted:', r)
                r = -10000
            
            reward_model.store_transition(tmp1, action, r, tmp2)
            agent.learn()
            reward_model.learn()
            observation = observation_
            pre_action = action
            steps += 1

        eps_history.append(agent.epsilon)
        avg_score = np.mean(scores[-100:])
        print('step:', i, 'last reward:', reward)
        # print('episode', i, 'score %.2f'% score, 'average score %.2f'% avg_score,
        #         'epsilon %.2f' % agent.epsilon)

    T.save(agent.Q_eval.state_dict(),\
           str(n_games)+'_iteration_reward_model_test.pt')
    
    # x = [i+1 for i in range(n_games)]
    filename = 'lunar_lander.png'
    # plotLearning(x, scores, eps_history, filename)
