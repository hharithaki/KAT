import numpy as np
import math
num_guards = 3
num_attacker = 3

def attacker_in_range(attacker_obs, A):
        for agent in enumerate(attacker_obs):
            if (agent[1][0]) == 1.0:
                b = np.array([[agent[1][1]],[agent[1][2]],[1]])
                x = svd_sol(A,b)
                if np.all(x>=0):
                    return(True)
        return(False)

def svd_sol(A, b):
    U, sigma, Vt = np.linalg.svd(A)
    sigma[sigma<1e-10] = 0
    sigma_reci = [(1/s if s!=0 else 0) for s in sigma]
    sigma_reci = np.diag(sigma_reci)
    x = Vt.transpose().dot(sigma_reci).dot(U.transpose()).dot(b)
    return(x)

def get_others_distance(partial_obs, x_pos, y_pos):
    distance_list = []
    agent_pos = np.array([x_pos,y_pos])
    for i in range(len(partial_obs)):
        if (partial_obs[i][0]) == 1.0:
            att_pos = np.array([float(partial_obs[i][1]),float(partial_obs[i][2])])
            dist_to_agent = np.sqrt(np.sum(np.square(att_pos-agent_pos)))
            distance_list.append(dist_to_agent)
        else:
            distance_list.append(10) # very high distance for dead 
    return distance_list

def get_nearest_agent_coordinates(partial_obs, x_pos, y_pos):
    other_distance = get_others_distance(partial_obs, x_pos, y_pos)
    min_index = other_distance.index(min(other_distance))
    return(partial_obs[min_index][1], partial_obs[min_index][2], min(other_distance))
    
def get_angle(x_oth, x_pos, y_oth, y_pos):
    deltaX = x_oth - x_pos
    deltaY = y_oth - y_pos
    rad = math.atan2(deltaY, deltaX)
    att_degree = math.degrees(rad)
    if att_degree < 0:
        att_degree = att_degree - abs(att_degree)
    return att_degree

def attacker_in_shoot_circle(partial_obs, x_pos, y_pos, ori):
    ori_degree = math.degrees(ori)
    while ori_degree > 360:
        ori_degree = ori_degree - 360
    for i in range(len(partial_obs)):
        if (partial_obs[i][0]) == 1.0:
            r2 = np.sqrt(np.square(x_pos - partial_obs[i][1]) + np.square(y_pos - partial_obs[i][2]))
            if r2 < 0.8:
                att_degree = get_angle(partial_obs[i][1], x_pos, partial_obs[i][2], y_pos)
                thetax = ori_degree - att_degree
                thetay = 360 - thetax
                if thetax < thetay:
                    action = 6
                elif thetax > thetay:
                    action = 5
                return True, action
    return False, None

# spread out and search attackers guard with rules
def guard_policy1(current_obs, agent_name, A):
    x_pos = current_obs[agent_name][1]
    y_pos = current_obs[agent_name][2]
    ori = current_obs[agent_name][3]
    can_shoot = attacker_in_range(current_obs[num_guards:,:], A)
    if can_shoot: # how to handle all the guards shooting the same attacker
        gua_action = 7
    else:
        # check for any attackers in shooting range if rotated
        rotate, action = attacker_in_shoot_circle(current_obs[num_guards:,:], x_pos, y_pos, ori)
        if rotate:
            gua_action = action
        else:
            # if near the walls/fort move away/towards
            if x_pos < -0.9: # too near to the left wall 
                # need to face that side to move? 
                # currently can move backwards
                gua_action = 1
            elif x_pos > 0.9: # too near to the right wall
                gua_action = 2
            elif y_pos > 0.75: # too near to the upper wall
                gua_action = 4
            elif y_pos < 0.2: # leaving the fort zone
                gua_action = 3
            else:
                # if all good find an attacker and get him to range
                x_val, y_val, dist = get_nearest_agent_coordinates(current_obs[num_guards:,:], x_pos, y_pos)
                if (y_val >= 0.2):
                    if (abs(x_val-x_pos) > abs(y_val-y_pos)):
                        if (x_val > x_pos):
                            gua_action = 1
                        else:
                            gua_action = 2
                    else:
                        if (y_val > y_pos):
                            gua_action = 3
                        else:
                            gua_action = 4
                else:
                    # if too close to other guards they can move away
                    guards_obs = current_obs[:num_guards,:]
                    guards_obs = np.delete(guards_obs, agent_name, axis=0)
                    x_gua, y_gua, dist = get_nearest_agent_coordinates(guards_obs, x_pos, y_pos)
                    # lets spread out only in x direction for now
                    if (abs(x_gua-x_pos) > 0.5):
                        if (abs(x_gua-x_pos) < 0.7):
                            gua_action = 0 # do nothing
                        else:
                            if (x_gua > x_pos):
                                gua_action = 1 # move toward
                            else:
                                gua_action = 2
                    else:
                        if (x_gua > x_pos):
                            gua_action = 2 # move away
                        else:
                            gua_action = 1
            # may be take this to the nearest using the 4 directions instead of degrees which can have 360 values
            # ori_degree = math.degrees(ori) if math.degrees(ori) < 360 else math.degrees(ori)-360
            # if gua_action == 1 and (ori_degree < 330 or ori_degree > 30):
            #     if ori_degree < 180:
            #         gua_action = 6
            #     else:
            #         gua_action = 5
            # elif gua_action == 2 and (ori_degree < 150 or ori_degree > 210):
            #     if 0 < ori_degree < 160:
            #         gua_action = 5
            #     else:
            #         gua_action = 6
            # elif gua_action == 3 and (ori_degree < 60 or ori_degree > 120):
            #     if 70 < ori_degree < 290:
            #         gua_action = 6
            #     else:
            #         gua_action = 5
            # elif gua_action == 4 and (ori_degree < 240 or ori_degree > 300):
            #     if 70 < ori_degree < 250:
            #         gua_action = 5
            #     else:
            #         gua_action = 6
    return gua_action

# no spread out but search attackers guard with rules
def guard_policy2(current_obs, agent_name, A):
    x_pos = current_obs[agent_name][1]
    y_pos = current_obs[agent_name][2]
    ori = current_obs[agent_name][3]
    can_shoot = attacker_in_range(current_obs[num_guards:,:], A)
    if can_shoot: # how to handle all the guards shooting the same attacker
        gua_action = 7
    else:
        # check for any attackers in shooting range if rotated
        rotate, action = attacker_in_shoot_circle(current_obs[num_guards:,:], x_pos, y_pos, ori)
        if rotate:
            gua_action = action
        else:
            # if near the walls/fort move away/towards
            if x_pos < -0.9: # too near to the left wall 
                # need to face that side to move? 
                # currently can move backwards
                gua_action = 1
            elif x_pos > 0.9: # too near to the right wall
                gua_action = 2
            elif y_pos > 0.75: # too near to the upper wall
                gua_action = 4
            elif y_pos < 0.2: # leaving the fort zone
                gua_action = 3
            else:
                # if all good find an attacker and get him to range
                x_val, y_val, dist = get_nearest_agent_coordinates(current_obs[num_guards:,:], x_pos, y_pos)
                if (y_val >= 0.2):
                    if (abs(x_val-x_pos) > abs(y_val-y_pos)):
                        if (x_val > x_pos):
                            gua_action = 1
                        else:
                            gua_action = 2
                    else:
                        if (y_val > y_pos):
                            gua_action = 3
                        else:
                            gua_action = 4
                else:
                    gua_action = 0 # do nothing
    return gua_action

# spread out search attackers guard with force fields
def guard_policy3(current_obs, agent_name, A):
    force_fort = 4
    force_other_walls = 2.8
    force_upper_wall = 0.28
    force_guard = 4
    x_pos = current_obs[agent_name][1]
    y_pos = current_obs[agent_name][2]
    ori = current_obs[agent_name][3]
    can_shoot = attacker_in_range(current_obs[num_guards:,:], A)
    if can_shoot: # how to handle all the guards shooting the same attacker
        gua_action = 7
    else:
        # check for any attackers in shooting range if rotated
        rotate, action = attacker_in_shoot_circle(current_obs[num_guards:,:], x_pos, y_pos, ori)
        if rotate:
            gua_action = action
        else:
            # force field moving and do nothing
            # force from fort
            angle = get_angle(0, x_pos, 0.8, y_pos)
            fx = force_fort*np.cos(np.radians(angle)) # > 90 negative else positive
            fy = force_fort*np.sin(np.radians(angle))

            # force from walls
            if (x_pos >= 0.8):
                fxw = -(force_other_walls - (1.0 - abs(x_pos))*14)
            elif (x_pos <= -0.8):
                fxw = force_other_walls - (1.0 - abs(x_pos))*14
            else:
                fxw = 0
            if (y_pos >= 0.6):
                fyw = -(force_upper_wall - (0.8 - abs(y_pos))*1.4)
            elif (y_pos <= -0.6):
                fyw = force_other_walls - (0.8 - abs(y_pos))*14
            else:
                fyw = 0

            # force from guards         
            guards_obs = current_obs[:num_guards,:]
            guards_obs = np.delete(guards_obs, agent_name, axis=0)
            x_gua, y_gua, dist = get_nearest_agent_coordinates(guards_obs, x_pos, y_pos)
            angle_gua = get_angle(x_gua, x_pos, y_gua, y_pos)
            if dist < 0.5:
                if (x_pos < x_gua):
                    fxg = -abs(force_guard*np.cos(np.radians(angle_gua)))
                else:
                    fxg = abs(force_guard*np.cos(np.radians(angle_gua)))
                if (y_pos < y_gua):
                    fyg = -abs(force_guard*np.sin(np.radians(angle_gua)))
                else:
                    fyg = abs(force_guard*np.sin(np.radians(angle_gua)))
            elif dist > 0.7:
                if (x_pos < x_gua):
                    fxg = abs(force_guard*np.cos(np.radians(angle_gua)))
                else:
                    fxg = -abs(force_guard*np.cos(np.radians(angle_gua)))
                if (y_pos < y_gua):
                    fyg = abs(force_guard*np.sin(np.radians(angle_gua)))
                else:
                    fyg = -abs(force_guard*np.sin(np.radians(angle_gua)))
            else:
                fxg = 0
                fyg = 0

            # force from attackers
            x_val, y_val, dist = get_nearest_agent_coordinates(current_obs[num_guards:,:], x_pos, y_pos)
            angle_att = get_angle(x_val, x_pos, y_val, y_pos)
            if (x_pos < x_val):
                fxa = abs(force_guard*np.cos(np.radians(angle_att)))
            else:
                fxa = -abs(force_guard*np.cos(np.radians(angle_att)))
            if (y_pos < y_val):
                fya = abs(force_guard*np.sin(np.radians(angle_att)))
            else:
                fya = -abs(force_guard*np.sin(np.radians(angle_att)))

            # total force
            total_fx = fx + fxw + fxg + fxa
            total_fy = fy + fyw + fyg + fya

            if abs(total_fx) > abs(total_fy):
                if total_fx > 0:
                    gua_action = 1
                else:
                    gua_action = 2
            elif (abs(total_fx) < abs(total_fy)):
                if total_fy > 0:
                    gua_action = 3
                else:
                    gua_action = 4
            else:
                gua_action = 0
    return gua_action

# no shooting no spread out attacker with force fields
def attacker_policy1(current_obs, agent_name):
    force_fort = 5
    force_other_walls = 2.8
    force_upper_wall = 0.28
    x_pos = current_obs[agent_name][1]
    y_pos = current_obs[agent_name][2]
    ori = current_obs[agent_name][3]
    # force from fort
    angle = get_angle(0, x_pos, 0.8, y_pos)
    fx = force_fort*np.cos(np.radians(angle)) # > 90 negative else positive
    fy = force_fort*np.sin(np.radians(angle))
    
    # force from walls
    if (x_pos >= 0.8):
        fxw = -(force_other_walls - (1.0 - abs(x_pos))*14)
    elif (x_pos <= -0.8):
        fxw = force_other_walls - (1.0 - abs(x_pos))*14
    else:
        fxw = 0
    if (y_pos >= 0.6):
        fyw = -(force_upper_wall - (0.8 - abs(y_pos))*1.4)
    elif (y_pos <= -0.6):
        fyw = force_other_walls - (0.8 - abs(y_pos))*14
    else:
        fyw = 0

    # total force
    total_fx = fx + fxw
    total_fy = fy + fyw

    if abs(total_fx) > abs(total_fy):
        if total_fx > 0:
            att_action = 1
        else:
            att_action = 2
    elif abs(total_fx) < abs(total_fy):
        if total_fy > 0:
            att_action = 3
        else:
            att_action = 4
    else:
        att_action = 0
    
    # ori_degree = math.degrees(ori) if math.degrees(ori) < 360 else math.degrees(ori)-360
    # if att_action == 1 and (ori_degree < 330 or ori_degree > 30):
    #     if ori_degree < 180:
    #         att_action = 6
    #     else:
    #         att_action = 5
    # elif att_action == 2 and (ori_degree < 150 or ori_degree > 210):
    #     if 0 < ori_degree < 160:
    #         att_action = 5
    #     else:
    #         att_action = 6
    # elif att_action == 3 and (ori_degree < 60 or ori_degree > 120):
    #     if 70 < ori_degree < 290:
    #         att_action = 6
    #     else:
    #         att_action = 5
    # elif att_action == 4 and (ori_degree < 240 or ori_degree > 300):
    #     if 70 < ori_degree < 250:
    #         att_action = 5
    #     else:
    #         att_action = 6
            
    return att_action

# no shooting but spread out attacker with force fields
def attacker_policy2(current_obs, agent_name):
    force_fort = 5
    force_other_walls = 2.8
    force_upper_wall = 0.28
    force_oth = 4
    x_pos = current_obs[agent_name][1]
    y_pos = current_obs[agent_name][2]

    # force from fort
    angle = get_angle(0, x_pos, 0.8, y_pos)
    fx = force_fort*np.cos(np.radians(angle)) # > 90 negative else positive
    fy = force_fort*np.sin(np.radians(angle))

    # force from walls
    if (x_pos >= 0.8):
        fxw = -(force_other_walls - (1.0 - abs(x_pos))*14)
    elif (x_pos <= -0.8):
        fxw = force_other_walls - (1.0 - abs(x_pos))*14
    else:
        fxw = 0
    if (y_pos >= 0.6):
        fyw = -(force_upper_wall - (0.8 - abs(y_pos))*1.4)
    elif (y_pos <= -0.6):
        fyw = force_other_walls - (0.8 - abs(y_pos))*14
    else:
        fyw = 0

    # force from other agents (all)         
    obs = np.delete(current_obs, agent_name, axis=0)
    x_oth, y_oth, dist = get_nearest_agent_coordinates(obs, x_pos, y_pos)
    angle_age = get_angle(x_oth, x_pos, y_oth, y_pos)
    if dist < 0.8: # change and see what happens
        if (x_pos < x_oth):
            fxg = -abs(force_oth*np.cos(np.radians(angle_age)))
        else:
            fxg = abs(force_oth*np.cos(np.radians(angle_age)))
        if (y_pos < y_oth):
            fyg = -abs(force_oth*np.sin(np.radians(angle_age)))
        else:
            fyg = abs(force_oth*np.sin(np.radians(angle_age)))
    else:
        fxg = 0
        fyg = 0

    # total force
    total_fx = fx + fxw + fxg
    total_fy = fy + fyw + fyg

    if abs(total_fx) > abs(total_fy):
        if total_fx > 0:
            att_action = 1
        else:
            att_action = 2
    elif abs(total_fx) < abs(total_fy):
        if total_fy > 0:
            att_action = 3
        else:
            att_action = 4
    else:
        att_action = 0
    return att_action

# shooting and spread out attacker with force fields
def attacker_policy3(current_obs, agent_name, A):
    force_fort = 5
    force_other_walls = 2.8
    force_upper_wall = 0.28
    force_oth = 4
    x_pos = current_obs[agent_name][1]
    y_pos = current_obs[agent_name][2]

    can_shoot = attacker_in_range(current_obs[:num_guards,:], A) # actually guard in range
    if can_shoot:
        att_action = 7
    else:
        # force from fort
        angle = get_angle(0, x_pos, 0.8, y_pos)
        fx = force_fort*np.cos(np.radians(angle)) # > 90 negative else positive
        fy = force_fort*np.sin(np.radians(angle))

        # force from walls
        if (x_pos >= 0.8):
            fxw = -(force_other_walls - (1.0 - abs(x_pos))*14)
        elif (x_pos <= -0.8):
            fxw = force_other_walls - (1.0 - abs(x_pos))*14
        else:
            fxw = 0
        if (y_pos >= 0.6):
            fyw = -(force_upper_wall - (0.8 - abs(y_pos))*1.4)
        elif (y_pos <= -0.6):
            fyw = force_other_walls - (0.8 - abs(y_pos))*14
        else:
            fyw = 0

        # force from other agents (all)         
        obs = np.delete(current_obs, agent_name, axis=0)
        x_oth, y_oth, dist = get_nearest_agent_coordinates(obs, x_pos, y_pos)
        angle_age = get_angle(x_oth, x_pos, y_oth, y_pos)
        if dist < 0.8: # change and see what happens
            if (x_pos < x_oth):
                fxg = -abs(force_oth*np.cos(np.radians(angle_age)))
            else:
                fxg = abs(force_oth*np.cos(np.radians(angle_age)))
            if (y_pos < y_oth):
                fyg = -abs(force_oth*np.sin(np.radians(angle_age)))
            else:
                fyg = abs(force_oth*np.sin(np.radians(angle_age)))
        else:
            fxg = 0
            fyg = 0

        # total force
        total_fx = fx + fxw + fxg
        total_fy = fy + fyw + fyg

        if abs(total_fx) > abs(total_fy):
            if total_fx > 0:
                att_action = 1
            else:
                att_action = 2
        elif abs(total_fx) < abs(total_fy):
            if total_fy > 0:
                att_action = 3
            else:
                att_action = 4
        else:
            att_action = 0
    return att_action

