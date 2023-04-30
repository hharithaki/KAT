import numpy as np
import math
import policies

class Policy:
    def __init__(self):
        self.wall_pos = [-1,1,-0.8,0.8]
        self.fortDim = 0.15   # radius
        self.doorLoc = np.array([0,0.8])
        # simulation timestep
        self.dt = 0.1
        self.max_speed = 3
        self.max_rot = 0.17
        self.size = 0.05
        self.shootRad = 0.8
        self.shootWin = np.pi/4
        
    def get_tri_pts_arr(self, x_pos, y_pos, ori):
        ang = ori
        pt1 = [x_pos, y_pos]+self.size*np.array([np.cos(ang), np.sin(ang)])
        pt2 = pt1 + self.shootRad*np.array([np.cos(ang+self.shootWin/2), np.sin(ang+self.shootWin/2)])
        pt3 = pt1 + self.shootRad*np.array([np.cos(ang-self.shootWin/2), np.sin(ang-self.shootWin/2)])
        
        A = np.array([[pt1[0], pt2[0], pt3[0]],
                      [pt1[1], pt2[1], pt3[1]],
                      [     1,      1,      1]])       
        return(A)             

    def get_actions(self, current_obs):
        actions = np.zeros(6, dtype=int)
        for i in range(len(current_obs)):
            if current_obs[i][0] != 1.0:
                actions[i] = 0 # agent dead - do nothing
            else:
                x_pos = current_obs[i][1]
                y_pos = current_obs[i][2]
                ori = current_obs[i][3]
                if i < 3:
                    A = self.get_tri_pts_arr(x_pos, y_pos, ori) # shoot cone
                    actions[i] = policies.guard_policy1(current_obs, i, A)
                else: # attacker
                    actions[i] = policies.attacker_policy1(current_obs, i)                
        return actions

    def get_other_agents_actions(self, obs):
        actions = np.zeros(6, dtype=int)
        for i in range(len(obs)):
            if obs[i][0] != 1.0:
                actions[i] = 0 # agent dead - do nothing
            else:
                x_pos = obs[i][1]
                y_pos = obs[i][2]
                ori = obs[i][3]
                # guards
                if i < 3:
                    A = self.get_tri_pts_arr(x_pos, y_pos, ori)
                    actions[i] = policies.guard_policy1(obs, i, A)
                else: # attacker
                    actions[i] = policies.attacker_policy1(obs, i)                
        return actions
