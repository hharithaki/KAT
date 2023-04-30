import numpy as np
from gym_fortattack.fortattack import make_fortattack_env 
from gym_fortattack.fortattack_test import make_testfortattack_env
                                                                                                                                                                                                                                                                                        
def normalize_obs(obs, mean, std):
    if mean is not None:
        return np.divide((obs - mean), std)
    else:
        return obs

def make_single_env(args):
    env = make_fortattack_env(args.num_env_steps)
    return(env)

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

def make_test_env(start_state):
    env = make_testfortattack_env(start_state)
    return env

def construct_data(obs, alive_ag):
    door = np.array([0,0.8])
    dist_to, alive_att, alive_gua = ([] for ia in range(3))
    x_pos_arr, y_pos_arr, rot_arr, x_vel_arr, y_vel_arr = ([] for ib in range (5))

    for j in range(len(obs)):
        x_pos_arr.append(obs[j][1])
        y_pos_arr.append(obs[j][2])
        rot_arr.append(obs[j][3])
        x_vel_arr.append(obs[j][4])
        y_vel_arr.append(obs[j][5])

    for i in range(len(obs)):
        arr = np.array([float(obs[i][1]),float(obs[i][2])])
        distToDoor = np.sqrt(np.sum(np.square(arr-door)))
        dist_to.append(distToDoor)  

    if len(alive_ag) == 0:
        alive_att = np.full(len(obs),2)
        alive_gua = np.full(len(obs),2)
    else:
        for k in range(len(obs)):
            alive_att.append(alive_ag[k][0])
            alive_gua.append(alive_ag[k][1])

    data = np.stack((x_pos_arr, y_pos_arr, rot_arr, x_vel_arr, y_vel_arr, alive_att, alive_gua, dist_to), axis=-1)

    return data

