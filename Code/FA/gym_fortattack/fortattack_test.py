import gym
from gym import spaces
import numpy as np
from multiagent.multi_discrete import MultiDiscrete
from malib.spaces import Box, MASpace,  MAEnvSpec

def make_testfortattack_env(start_state):
    scenario = gym.make('fortattack-v2', observation=start_state)
    world = scenario.world
    world.max_time_steps = 1000
    env = FortAttackGlobalEnv(world, scenario.reward, scenario.observation)
    return env

class FortAttackGlobalEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array']
    }

    def terminate(self):
        pass

    def __init__(self, world, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None, shared_viewer=True):
        self.ob_rms = None
        self.world = world
        self.agents = self.world.policy_agents
        self.n = len(world.policy_agents)
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback
        # environment parameters
        self.discrete_action_space = True
        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = True #False
        # if true, even the action is continuous, action will be performed discretely
        self.force_discrete_action = world.discrete_action if hasattr(world, 'discrete_action') else False
        # if true, every agent has the same reward
        self.shared_reward = world.collaborative if hasattr(world, 'collaborative') else False

        # configure spaces
        self.action_space = []
        self.observation_space = []
        obs_shapes = []
        self.agent_num = len(self.agents)
        for agent in self.agents:
            total_action_space = []
            # physical action space
            if self.discrete_action_space:
                u_action_space = spaces.Discrete((world.dim_p) * 2 + 2)   ##
            else:
                u_action_space = spaces.Box(low=-agent.u_range, high=+agent.u_range, shape=(world.dim_p,), dtype=np.float32)
            if agent.movable:
                total_action_space.append(u_action_space)
            # communication action space
            if self.discrete_action_space:
                c_action_space = spaces.Discrete(world.dim_c)
            else:
                c_action_space = spaces.Box(low=0.0, high=1.0, shape=(world.dim_c,), dtype=np.float32)
            if not agent.silent:
                total_action_space.append(c_action_space)
            # total action space
            if len(total_action_space) > 1:
                # all action spaces are discrete, so simplify to MultiDiscrete action space
                if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]):
                    act_space = MultiDiscrete([[0, act_space.n - 1] for act_space in total_action_space])
                else:
                    act_space = spaces.Tuple(total_action_space)
                self.action_space.append(act_space)
            else:
                self.action_space.append(total_action_space[0])
            # observation space
            obs_dim = len(observation_callback(agent, self.world))
            obs_shapes.append((obs_dim,))
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32))
            agent.action.c = np.zeros(self.world.dim_c)

        # action has 8 values: 
        # nothing, +forcex, -forcex, +forcey, -forcey, +rot, -rot, shoot 
        self.action_spaces = MASpace(tuple(Box(low=0., high=1., shape=((world.dim_p) * 2 + 2,)) for _ in range(self.agent_num)))  ##
        self.observation_spaces = MASpace(tuple(Box(low=-np.inf, high=+np.inf, shape=obs_shape) for obs_shape in obs_shapes))

        self.env_specs = MAEnvSpec(self.observation_spaces, self.action_spaces)
        self.action_range = [0., 1.]

    def step(self, action_n):
        obs_n = []
        done_n = []
        alive_g = []
        info_n = {'n': []}
        self.agents = self.world.policy_agents
        for i, agent in enumerate(self.agents):
            action = action_n[i]
            self._set_action(action, agent, self.action_space[i]) # sets the actions in the agent object
        
        # advance world state
        self.world.step()
        
        # record observation for each agent
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
            alive_g.append([self.world.numAliveAttackers, self.world.numAliveGuards]) 
            info_n['n'].append(self._get_info(agent))

        ## implement single done reflecting game state
        done = self._get_done()
        self.world.time_step += 1
        obs_n = np.array(obs_n)      
        alive_g = np.array(alive_g)
        return obs_n, done, alive_g, info_n

    # def reset(self):
    #     # reset world
    #     self.reset_callback()
    #     # record observations for each agent
    #     obs_n = []
    #     self.agents = self.world.policy_agents
    #     for agent in self.agents:
    #         obs_n.append(self._get_obs(agent))
    #     obs_n = np.array(obs_n)
    #     return obs_n

    # get info used for benchmarking
    def _get_info(self, agent):
        if self.info_callback is None:
            return {}
        return self.info_callback(agent, self.world)

    # get observation for a particular agent
    def _get_obs(self, agent):
        if self.observation_callback is None:
            return np.zeros(0)
        return self.observation_callback(agent, self.world)

    # get done for the whole environment
    # unused right now -- agents are allowed to go beyond the viewing screen
    def _get_done(self):
        # done if any attacker reached landmark, attackers win
        th = self.world.fortDim
        for attacker in self.world.alive_attackers:
            dist = np.sqrt(np.sum(np.square(attacker.state.p_pos-self.world.doorLoc)))
            if dist < th:
                # print('attacker reached fort')
                self.world.gameResult[2] = 1
                return(True)

        # done if all attackers are dead, guards win
        if self.world.numAliveAttackers == 0:
            # print('all attackers dead')
            self.world.gameResult[0] = 1
            return(True)    
        
        # done if max number of time steps over, guards win
        elif self.world.time_step == self.world.max_time_steps-1:
            # print('max number of time steps')
            self.world.gameResult[1] = 1
            return(True)

        # otherwise not done
        return False

    # set env action for a particular agent
    def _set_action(self, action, agent, action_space, time=None):
        agent.action.u = np.zeros(self.world.dim_p)
        agent.action.c = np.zeros(self.world.dim_c)
        # process action
        if isinstance(action_space, MultiDiscrete):
            act = []
            size = action_space.high - action_space.low + 1
            index = 0
            for s in size:
                act.append(action[index:(index+s)])
                index += s
            action = act
        else:
            action = [action]

        if agent.movable:
            # print('self.discrete_action_input', self.discrete_action_input) # True
            # physical action
            if self.discrete_action_input:
                agent.action.u = np.zeros(self.world.dim_p)     ## We'll use this now for Graph NN
                # process discrete action
                ## if action[0] == 0, then do nothing
                if action[0] == 1: agent.action.u[0] = +1.0
                if action[0] == 2: agent.action.u[0] = -1.0
                if action[0] == 3: agent.action.u[1] = +1.0
                if action[0] == 4: agent.action.u[1] = -1.0
                if action[0] == 5: agent.action.u[2] = +agent.max_rot
                if action[0] == 6: agent.action.u[2] = -agent.max_rot
                agent.action.shoot = True if action[0] == 7 else False

            else:
                if self.force_discrete_action:       
                    d = np.argmax(action[0])
                    action[0][:] = 0.0
                    action[0][d] = 1.0
                if self.discrete_action_space:      ## this was begin used in PR2 Paper
                    # print('action', action)
                    agent.action.u[0] += action[0][1] - action[0][2]    ## each is 0 to 1, so total is -1 to 1
                    agent.action.u[1] += action[0][3] - action[0][4]    ## same as above
                    
                    ## simple shooting action
                    agent.action.shoot = True if action[0][6]>0.5 else False   # a number greater than 0.5 would mean shoot

                    ## simple rotation model
                    agent.action.u[2] = 2*(action[0][5]-0.5)*agent.max_rot
            
                else:
                    agent.action.u = action[0]
            sensitivity = 5.0   # default if no value specified for accel
            if agent.accel is not None:
                sensitivity = agent.accel
            agent.action.u[:2] *= sensitivity
            
            ## remove used actions
            action = action[1:]
        

        if not agent.silent:
            # communication action
            if self.discrete_action_input:
                agent.action.c = np.zeros(self.world.dim_c)
                agent.action.c[action[0]] = 1.0
            else:
                agent.action.c = action[0]
            action = action[1:]
        
        # make sure we used all elements of action
        assert len(action) == 0