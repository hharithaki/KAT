import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym_fortattack.core import World, Agent, EntityState
import numpy as np
import time

class FortAttackEnvV2(gym.Env):  
    metadata = {'render.modes': ['human']}   
    def __init__(self, observation=None):
        self.world = World() 

        self.world.fortDim = 0.15
        self.world.doorLoc = np.array([0,0.8])
        tempGuards = 0
        tempAttackers = 0
        for i in range(len(observation)):
            if(i < 2 and observation[i][0] == 1.0):
                tempGuards += 1
            elif(i > 1 and observation[i][0] == 1.0):
                tempAttackers += 1
        
        self.world.numGuards = 3
        self.world.numAttackers = 3
        self.world.numBullets = 0 # can be removed??
        self.world.bullets = []
        self.world.numAgents = self.world.numGuards + self.world.numAttackers
        self.world.numAliveGuards, self.world.numAliveAttackers, self.world.numAliveAgents = tempGuards, tempAttackers, (tempGuards+tempAttackers)
        landmarks = [] # as of now no obstacles, landmarks 

        self.world.agents = [Agent() for i in range(self.world.numAgents)]
        for i, agent in enumerate(self.world.agents):
            agent.name = 'agent %d' % (i+1)
            agent.collide = True
            agent.silent = True
            agent.attacker = False if i < self.world.numGuards else True
            agent.accel = 3  ## guards and attackers have same speed and accel
            agent.max_speed = 3   ## used in integrate_state() inside core.py. slowing down so that bullet can move fast and still it doesn't seem that the bullet is skipping steps 
            agent.max_rot = 0.17 ## approx 10 degree
            
            agent.color = np.array([0.0, 1.0, 0.0]) if not agent.attacker else np.array([1.0, 0.0, 0.0])
            agent.state.p_vel = np.ndarray(2, buffer = np.array([observation[i][4], observation[i][5]]))
            agent.state.c = np.zeros(self.world.dim_c)
            agent.state.p_ang = observation[i][3]
            xMin, xMax, yMin, yMax = self.world.wall_pos
            agent.state.p_pos = np.ndarray(2, buffer=np.array([observation[i][1], observation[i][2]]))
            
            if (observation[i][0] == 1.0):
                agent.alive = True
                agent.numHit = 0 # overall in one episode
                agent.numWasHit = 0
                agent.hit = False # in last time step
                agent.wasHit = False
            else:
                agent.alive = False
                agent.numHit = 1 # overall in one episode
                agent.numWasHit = 1
                agent.hit = True # in last time step
                agent.wasHit = True
            
        self.viewers = [None]
        self.render_geoms = None
        self.shared_viewer = True
        self.world.time_step = 0
        self.world.max_time_steps = None
        self.world.vizDead = False
        self.world.vizAttn = True
        self.world.gameResult = np.array([0,0,0]) #  [all attackers dead, max time steps, attacker reached fort]            

    def reward(self, agent):
        if agent.alive or agent.justDied:
            main_reward = self.attacker_reward(agent) if agent.attacker else self.guard_reward(agent)
        else:
            main_reward = 0
        return main_reward

    def attacker_reward(self, agent):
        rew0, rew1, rew2, rew3, rew4, rew5 = 0,0,0,0,0,0
        
        # dead agents are not getting reward just when they are dead
        # # Attackers get reward for being close to the door
        distToDoor = np.sqrt(np.sum(np.square(agent.state.p_pos-self.world.doorLoc)))
        if agent.prevDist is not None:
            rew0 = 2*(agent.prevDist - distToDoor)
            # print('rew0', rew0, 'fortattack_env_v1.py')
        # Attackers get very high reward for reaching the door
        th = self.world.fortDim
        if distToDoor<th:
            rew1 = 10
            self.world.atttacker_reached = True

        # attacker gets -ve reward for using laser
        if agent.action.shoot:
            rew2 = -1

        # gets positive reward for hitting a guard??
        if agent.hit:
            rew3 = +3

        # gets negative reward for being hit by a guard
        if agent.wasHit:
            rew4 = -3

        # high negative reward if all attackers are dead
        if self.world.numAliveAttackers == 0:
            rew5 = -10

        rew = rew0+rew1+rew2+rew3+rew4+rew5
        agent.prevDist = distToDoor.copy()
        # print('attacker_reward', rew1, rew2, rew3, rew4, rew)
        return rew

    def guard_reward(self, agent):
        # guards get reward for keeping all attacker away
        rew0, rew1, rew2, rew3, rew4, rew5, rew6, rew7, rew8 = 0,0,0,0,0,0,0,0,0
        
        # # high negative reward for leaving the fort
        selfDistToDoor = np.sqrt(np.sum(np.square(agent.state.p_pos-self.world.doorLoc)))
        # if selfDistToDoor>0.3:
        #     rew0 = -2

        # negative reward for going away from the fort
        if agent.prevDist is not None:
            if selfDistToDoor>0.3 and agent.prevDist<=0.3:
                rew0 = -1
            elif selfDistToDoor<=0.3 and agent.prevDist>0.3:
                rew0 = 1

            # rew1 = 20*(agent.prevDist - selfDistToDoor)

            # print('rew1', rew1, 'fortattack_env_v1.py')
        # rew1 = -0.1*selfDistToDoor

        # negative reward if attacker comes closer
        # make it exponential
        if self.world.numAliveAttackers != 0:
            minDistToDoor = np.min([np.sqrt(np.sum(np.square(attacker.state.p_pos-self.world.doorLoc))) for attacker in self.world.alive_attackers])
            # protectionRadius = 0.5
            # sig = protectionRadius/3
            # rew2 = -10*np.exp(-(minDistToDoor/sig)**2)

            # high negative reward if attacker reaches the fort
            th = self.world.fortDim
            if minDistToDoor<th:
                rew3 = -10

        # guard gets negative reward for using laser
        if agent.action.shoot:
            rew4 = -0.1

        # gets reward for hitting an attacker
        if agent.hit:
            rew5 = 3


        # guard gets -ve reward for being hit by laser
        if agent.wasHit:
            rew6 = -3

        # high positive reward if all attackers are dead
        if self.world.numAliveAttackers == 0:
            # if agent.hit:
            rew7 = 10

        # # small positive reward at every time step
        # rew8 = 10/self.world.max_time_steps

        rew = rew0+rew1+rew2+rew3+rew4+rew5+rew6+rew7+rew8
        # print('guard_reward', rew1, rew2, rew3, rew4, rew)
        agent.prevDist = selfDistToDoor.copy()
        return rew


    def observation(self, agent, world):
        entity_pos = []
        for entity in world.landmarks:
            if not entity.boundary:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        
        orien = [[agent.state.p_ang]]
        
        return(np.concatenate([[agent.alive]]+[agent.state.p_pos] + orien + [agent.state.p_vel] +entity_pos))

    
