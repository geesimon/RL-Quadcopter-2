import numpy as np
import sys
from task import MoveToTask
from agents.ddpg_agent import DDPGAgent

num_episodes = 100
target_pos = np.array([100., 100., 100.])
task = MoveToTask(target_pos=target_pos)
agent = DDPGAgent(task)
rewards = np.zeros(num_episodes)
positions = np.zeros((num_episodes, 3))
path = None

for i_episode in range(num_episodes):
    state = agent.reset_episode() # start a new episode
    while True:
        action = agent.act(state)
        next_state, reward, done = task.step(action)
        #print(reward)
        agent.step(action, reward, next_state, done)
        state = next_state
        if done:
            rewards[i_episode],  path = agent.get_score()
            dist = task.get_distance(task.sim.pose[0:3], task.target_pos)
            print("\rEpisode = {:4d}, score = {:7.3f} distance = {:7.3f}".format(i_episode + 1, 
                  rewards[i_episode], dist), end="")
            break
    sys.stdout.flush()