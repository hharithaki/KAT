import datetime
from operator import index
import numpy as np
import torch
import utils
import random
import time
from arguments import get_args
import action_policy
import ad_hoc
import csv
import sklweka.jvm as jvm

def m_fortattack(args):
    env = utils.make_single_env(args)
    obs = env.reset()
    act = action_policy.Policy()

    alive_ag = []
    jvm.start()
    start = time.time()
    for j in range(10):
        done = False
        step = 0
        previous_action = np.zeros((6,2))
        agent_actions = np.zeros(6) # set to zeros for previous action coutn
        learner_actions = []
        while not done:                    
            actions_list = act.get_actions(obs)
            agent_actions = np.array(actions_list).reshape(-1)
            # first agent = ad hoc agent
            if len(learner_actions) == 0:
                learner_actions = ad_hoc.learner(obs, alive_ag, previous_action, step)
                
            
            agent_actions[0] = learner_actions[0]
            obs, reward, done, alive_ag, info = env.step(agent_actions)
            learner_actions.pop(0)
            env.render()
            for agent in range(6):
                previous_action[agent][0] = previous_action[agent][1]
                previous_action[agent][1] = agent_actions[agent]
                
            step += 1 
            if done:
                obs = env.reset()
                masks = torch.FloatTensor(obs[:,0]) #check agents alive or dead
                time.sleep(1)
    jvm.stop()
    end = time.time()
if __name__ == '__main__':
    args = get_args()
    if args.seed is None:
        args.seed = random.randint(0,10000)
    torch.manual_seed(args.seed)    # set the random seed
    torch.set_num_threads(1)        # only one threah.
    np.random.seed(args.seed)

    m_fortattack(args)
    
