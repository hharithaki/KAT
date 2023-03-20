import math
import re
import subprocess
import numpy as np
from skspatial.objects import Line
from skspatial.objects import LineSegment
try:
  import hfo
except ImportError:
  print('Failed to import hfo. To install hfo, in the HFO directory'\
    ' run: \"pip install .\"')
  exit()

def process_answer(answer_list):
    answer = answer_list.rstrip().split('\n\n')
    if len(answer) > 1: # have more than 1 answer sets
        # if any of the answer sets has a answer with shoot first select that
        for item in answer:
            temp_split = item.strip('{}\n').split(', ')
            if len(temp_split) == 1 and 'occurs(shoot(learner),0)' in temp_split:
                return item
            elif 'occurs(shoot(learner),0)' in temp_split:
                pass
            else:
                return item
    else:
        return answer[0]

# Map actions with the fort-attack env action numbers
def map_actions(action_list, agent, teammates, opponents): # what about future predicts
    safe_index, velocity = distance_to_opponent([agent.x_pos,agent.y_pos], teammates, opponents)
    actions = []
    for element in action_list:
        if element.startswith('occurs(shoot('): # occurs(shoot(learner),2)
            action = hfo.SHOOT
        elif element.startswith('occurs(dribble('): # occurs(dribble(learner,5,6),0)
            action = hfo.DRIBBLE
        elif element.startswith('occurs(pass('): # occurs(pass(learner,offense2),0)
            if safe_index == 'PASS':
                action = (hfo.DRIBBLE) # pass to nearest
            else:
                action = (hfo.KICK_TO, teammates[safe_index].x_pos, teammates[safe_index].y_pos, velocity)
        actions.append(action)
    return actions

# Get distance from agent to nearest opponenet => can be replaced by procimity to nearest opponent
def distance_to_opponent(agent, teammates, opponenets):
    offense, defense = get_offense_defense_processed(teammates, opponenets)
    can_pass = []
    velocity_array = []
    for i in range(len(offense)):
        can_pass.append(True)
        velocity_array.append([])
    choosen = False
    opponent_max_velocity = 1
    for i in range(len(offense)):
        teammate = offense[i]
        for opponent in defense:
            # dist = abs((teammate[0]-agent[0])*(agent[1]-opponent[1])-(agent[0]-opponent[0])*(teammate[1]-agent[1]))/math.sqrt(((teammate[0]-agent[0])**2+(teammate[1]-agent[1])**2))
            line = Line.from_points(teammate, agent)
            line_segment = LineSegment(teammate, agent)
            # distance
            distance = line.distance_point(opponent)
            # project
            projection_point = line.project_point(opponent)
            if line_segment.contains_point(projection_point):
                time = distance/opponent_max_velocity
                side_dist = np.linalg.norm(np.asarray(agent)-np.asarray(opponent))
                dist_to_travel = math.sqrt(side_dist**2-distance**2)
                velocity_req = dist_to_travel/time
                if velocity_req >= 2:
                    can_pass[i] = False # cannot achive the pass
                else:
                    velocity_array[i].append(velocity_req)
    if any(val == True for val in can_pass):
        if all(val == True for val in can_pass):
            # find the best teammate to pass => the one towards the goal
            goal_center = np.array([0.9, 0])
            goal_proximity = []
            for i in range(len(offense)):
                if len(velocity_array[i]) == 0:
                    # no opponenets in that side - choose
                    index = i
                    choosen = True
                else:
                    proximity_goal = np.linalg.norm(np.asarray([offense[i]])-goal_center)
                    goal_proximity.append(proximity_goal)
            if not choosen:
                index = goal_proximity.index(min(goal_proximity))
        else:
            index = can_pass.index(True)
        if len(velocity_array[index]) != 0:
            velocity = 1.7 if max(velocity_array[index]) > 1.7 else 2
        else:
            velocity = 1.7
    else:
        # dribble away - random pass not good
        index = 'PASS'
        velocity = 0
    return index, velocity

# Process teammates and opponents such that only grid values are added
def get_offense_defense_processed(teammates, opponents):
    offense = []
    defense = []
    for teammate in teammates:
        offense.append([teammate.x_pos, teammate.y_pos])
    for opponent in opponents:
        defense.append([opponent.x_pos, opponent.y_pos])
    return offense, defense

# Get ASP terms for the ad hoc agent
def get_ASP_terms(input, agent_name, step, grid, ball_grid, able_to_kick):
    input.append('holds(in('+agent_name+','+str(grid[0])+','+str(grid[1])+'),'+str(step)+').')
    # input.append('holds(ball_in('+str(ball_grid[0])+','+str(ball_grid[1])+'),'+str(step)+').')
    input.append('holds(ball_in('+str(grid[0])+','+str(grid[1])+'),'+str(step)+').')
    if able_to_kick == 1.0:
        input.append('holds(has_ball(learner),'+str(step)+').') # agent has the ball
    else:
        input.append('-holds(has_ball(learner),'+str(step)+').')
    return input
    
# Get ASP terms for the teammates and opponents
def get_ASP_terms_other(input, step, agent, teammates, opponents):
    # for agent in teammates: currently do this only for the nearest teammate
    input.append('holds(agent_in(offense2,'+str(teammates[0].grid[0])+','+str(teammates[0].grid[1])+'),'+str(step)+').')
    for i in range(len(opponents)):
        if agent.grid[0] < 13: # cannot shoot
            if (agent.grid[0]-1 <= opponents[i].grid[0] <= agent.grid[0]+1) and (agent.grid[1]-1 <= opponents[i].grid[1] <= agent.grid[1]+1):
                input.append('holds(agent_in(defense'+str(i+1)+','+str(opponents[i].grid[0])+','+str(opponents[i].grid[1])+'),'+str(step)+').')
        else:
            input.append('holds(agent_in(defense'+str(i+1)+','+str(opponents[i].grid[0])+','+str(opponents[i].grid[1])+'),'+str(step)+').')
    return input

def get_future_ASP_terms(input, step, agent, locations, uniform_numbers, teammate_uniform_num):
    # [[x_A,y_A], [xt1,yt1], [xt2,yt2], [xdg,ydg], [xd1,yd1], [xd2,yd2]] 7.0 8.0 1.0 2.0 3.0
    # nearest teammate
    idx = uniform_numbers.index(teammate_uniform_num)
    grid_nearest_teammate = get_gridno(locations[idx][0], locations[idx][1])
    input.append('holds(agent_in(offense2,'+str(grid_nearest_teammate[0])+','+str(grid_nearest_teammate[1])+'),'+str(step)+').')
    opponents = process_locations_to_opponents(locations, uniform_numbers, True)
    for i in range(len(opponents)):
        if agent.grid[0] < 13: # cannot shoot
            if (agent.grid[0]-1 <= opponents[i][0] <= agent.grid[0]+1) and (agent.grid[1]-1 <= opponents[i][1] <= agent.grid[1]+1):
                input.append('holds(agent_in(defense'+str(i+1)+','+str(opponents[i][0])+','+str(opponents[i][1])+'),'+str(step)+').')
        else:
            input.append('holds(agent_in(defense'+str(i+1)+','+str(opponents[i][0])+','+str(opponents[i][1])+'),'+str(step)+').')
    return input

# Run environment
def run_environment(state, actions):
    uniform_numbers = []
    locations = []
    for idx in range(len(actions)):
        if idx < 4: # adhoc + teammate
            if idx == 0: # adhoc
                if actions[idx] == '8':
                    new_loc_adh = get_new_location(state[0], state[1], state[3], state[4], 0.024, 0.021)
                elif actions[idx] == '11':
                    new_loc_adh = get_new_location(state[0], state[1], 0.9, 0, 0.1, 0.052)
                else:
                    new_loc_adh = [state[0], state[1]]
                uniform_numbers.append(11.0)
                locations.append(new_loc_adh)
            elif idx == 1 and state[21] != -2: #t1
                if actions[idx] == '8':
                    new_loc_t1 = get_new_location(state[19], state[20], state[3], state[4], 0.024, 0.021)
                elif actions[idx] == '11':
                    new_loc_t1 = get_new_location(state[19], state[20], 0.9, 0, 0.1, 0.052)
                elif actions[idx] == '(5, 0.4, 0.3)':
                    new_loc_t1 = get_new_location(state[19], state[20], 0.4, 0.3, 0.052, 0.044)
                elif actions[idx] == '(5, 0.4, -0.3)':
                    new_loc_t1 = get_new_location(state[19], state[20], 0.4, -0.3, 0.052, 0.044)
                else:
                    new_loc_t1 = [state[19], state[20]]
                uniform_numbers.append(state[21])
                locations.append(new_loc_t1)
            elif idx == 2 and state[24] != -2: #t2
                if actions[idx] == '8':
                    new_loc_t2 = get_new_location(state[22], state[23], state[3], state[4], 0.024, 0.021)
                elif actions[idx] == '11':
                    new_loc_t2 = get_new_location(state[22], state[23], 0.9, 0, 0.1, 0.052)
                elif actions[idx] == '(5, 0.4, 0.3)':
                    new_loc_t2 = get_new_location(state[22], state[23], 0.4, 0.3, 0.052, 0.044)
                elif actions[idx] == '(5, 0.4, -0.3)':
                    new_loc_t2 = get_new_location(state[22], state[23], 0.4, -0.3, 0.052, 0.044)
                else:
                    new_loc_t2 = [state[22], state[23]]
                uniform_numbers.append(state[24])
                locations.append(new_loc_t2)
            elif idx == 3 and state[27] != -2: #t3
                if actions[idx] == '8':
                    new_loc_t3 = get_new_location(state[25], state[26], state[3], state[4], 0.024, 0.021)
                elif actions[idx] == '11':
                    new_loc_t3 = get_new_location(state[25], state[26], 0.9, 0, 0.1, 0.052)
                elif actions[idx] == '(5, 0.4, 0.3)':
                    new_loc_t3 = get_new_location(state[25], state[26], 0.4, 0.3, 0.052, 0.044)
                elif actions[idx] == '(5, 0.4, -0.3)':
                    new_loc_t3 = get_new_location(state[25], state[26], 0.4, -0.3, 0.052, 0.044)
                else:
                    new_loc_t3 = [state[25], state[26]]
                uniform_numbers.append(state[27])
                locations.append(new_loc_t3)
        else: # defense
            if idx == 4 and state[30] != -2: #goalie  
                if actions[idx] == '(5, -0.75, -0.175)':
                    new_loc_goa = get_new_location(state[28], state[29], -0.75, -0.175, 0.003, 0.006)
                elif actions[idx] == '(5, -0.65, 0)':
                    new_loc_goa = get_new_location(state[28], state[29], -0.65, 0, 0.01, 0.01)
                elif actions[idx] == '(7,)':
                    new_loc_goa = get_new_location(state[28], state[29], state[3], state[4], 0.040, 0.034)
                elif actions[idx] == '(5, -0.75, 0.175)':
                    new_loc_goa = get_new_location(state[28], state[29], -0.75, 0.175, 0.005, 0.009)
                else:
                    new_loc_goa = [state[28], state[29]]
                uniform_numbers.append(state[30])
                locations.append(new_loc_goa)
            elif idx == 5 and state[33] != -2: #d1
                if actions[idx] == '8':
                    new_loc_d1 = get_new_location(state[31], state[32], state[3], state[4], 0.070, 0.036)
                elif actions[idx] == '16':
                    goal_sorted_list = get_sorted_opponents(state, 5, 3, pos_x=state[3], pos_y=state[4])
                    new_loc_d1 = get_new_location(state[31], state[32], goal_sorted_list[0][1], goal_sorted_list[0][2], 0.070, 0.036)
                elif actions[idx] == '15':
                    goal_sorted_list = get_sorted_opponents(state, 5, 3, pos_x=0.9, pos_y=0)
                    new_loc_d1 = get_new_location(state[31], state[32], goal_sorted_list[0][1], goal_sorted_list[0][2], 0.070, 0.036)
                else:
                    new_loc_d1 = [state[31], state[32]]
                uniform_numbers.append(state[33])
                locations.append(new_loc_d1)
            elif idx == 6 and state[36] != -2: #d2
                if actions[idx] == '8':
                    new_loc_d2 = get_new_location(state[34], state[35], state[3], state[4], 0.070, 0.036)
                elif actions[idx] == '16':
                    goal_sorted_list = get_sorted_opponents(state, 5, 3, pos_x=state[3], pos_y=state[4])
                    new_loc_d2 = get_new_location(state[34], state[35], goal_sorted_list[0][1], goal_sorted_list[0][2], 0.070, 0.036)
                elif actions[idx] == '15':
                    goal_sorted_list = get_sorted_opponents(state, 5, 3, pos_x=0.9, pos_y=0)
                    new_loc_d2 = get_new_location(state[34], state[35], goal_sorted_list[0][1], goal_sorted_list[0][2], 0.070, 0.036)
                else:
                    new_loc_d2 = [state[34], state[35]]
                uniform_numbers.append(state[36])
                locations.append(new_loc_d2)
            elif idx == 7 and state[39] != -2: #d3
                if actions[idx] == '8':
                    new_loc_d3 = get_new_location(state[37], state[38], state[3], state[4], 0.070, 0.036)
                elif actions[idx] == '16':
                    goal_sorted_list = get_sorted_opponents(state, 4, 4, pos_x=state[3], pos_y=state[4])
                    new_loc_d3 = get_new_location(state[37], state[38], goal_sorted_list[0][1], goal_sorted_list[0][2], 0.070, 0.036)
                elif actions[idx] == '15':
                    goal_sorted_list = get_sorted_opponents(state, 4, 4, pos_x=0.9, pos_y=0)
                    new_loc_d3 = get_new_location(state[37], state[38], goal_sorted_list[0][1], goal_sorted_list[0][2], 0.070, 0.036)
                else:
                    new_loc_d3 = [state[37], state[38]]
                uniform_numbers.append(state[39])
                locations.append(new_loc_d3)
            elif idx == 8 and state[42] != -2: #d4
                if actions[idx] == '8':
                    new_loc_d4 = get_new_location(state[40], state[41], state[3], state[4], 0.070, 0.036)
                elif actions[idx] == '16':
                    goal_sorted_list = get_sorted_opponents(state, 4, 4, pos_x=state[3], pos_y=state[4])
                    new_loc_d4 = get_new_location(state[40], state[41], goal_sorted_list[0][1], goal_sorted_list[0][2], 0.070, 0.036)
                elif actions[idx] == '15':
                    goal_sorted_list = get_sorted_opponents(state, 4, 4, pos_x=0.9, pos_y=0)
                    new_loc_d4 = get_new_location(state[40], state[41], goal_sorted_list[0][1], goal_sorted_list[0][2], 0.070, 0.036)
                else:
                    new_loc_d4 = [state[40], state[41]]
                uniform_numbers.append(state[42])
                locations.append(new_loc_d4)
    return locations, uniform_numbers

def get_new_location(x, y, xa, ya, diff_x, diff_y):
    if x < xa and y < ya:
        new_loc = [x+diff_x, y+diff_y]
    elif x < xa and y > ya:
        new_loc = [x+diff_x, y-diff_y]
    elif x > xa and y < ya:
        new_loc = [x-diff_x, y+diff_y]
    elif x > xa and y > ya:
        new_loc = [x-diff_x, y-diff_y]
    else:
        new_loc = [x,y]
    return new_loc

def get_sorted_opponents(state, num_opponents, num_teammates, pos_x, pos_y): # get offense agents sorted list
  unum_list = []
  # ad ho agent unum 11
  opp_pos_x = state[0]
  opp_pos_y = state[1]
  dist = get_dist_normalized(pos_x, pos_y, opp_pos_x, opp_pos_y)
  unum_list.append(tuple([dist, opp_pos_x, opp_pos_y, 11]))

  # teamamte 1
  opp_pos_x = state[19]
  opp_pos_y = state[20]
  dist = get_dist_normalized(pos_x, pos_y, opp_pos_x, opp_pos_y)
  unum_list.append(tuple([dist, opp_pos_x, opp_pos_y, state[21]]))

  # teamamte 2
  opp_pos_x = state[22]
  opp_pos_y = state[23]
  dist = get_dist_normalized(pos_x, pos_y, opp_pos_x, opp_pos_y)
  unum_list.append(tuple([dist, opp_pos_x, opp_pos_y, state[24]]))

  # teamamte 3
  opp_pos_x = state[25]
  opp_pos_y = state[26]
  dist = get_dist_normalized(pos_x, pos_y, opp_pos_x, opp_pos_y)
  unum_list.append(tuple([dist, opp_pos_x, opp_pos_y, state[27]]))

  if len(unum_list) > 1:
    return sorted(unum_list, key=lambda x: x[0])
  return unum_list

def get_dist_normalized(ref_x, ref_y, src_x, src_y):
  return math.sqrt(math.pow((ref_x - src_x),2) +
                   math.pow(((68/52.5)*(ref_y - src_y)),2))

def process_data(state, agent_name):
    state = np.array(state, dtype=np.float64)
    values = []
    if agent_name < 3: # offense agent
        values.append(state[0]) # AHA x
        values.append(state[1]) # AHA y
        values.append(state[8]) # AHA goal open angel
        values.append(state[9]) # AHA distance to nearest opponent
        values.append(state[19]) # t1 x
        values.append(state[20]) # t1 y
        values.append(state[10]) # t1 goal open angel
        values.append(state[13]) # t1 distance to nearest opp
        values.append(state[22]) # t2 x
        values.append(state[23]) # t2 y
        values.append(state[11]) # t2 goal open angel
        values.append(state[14]) # t2 distance to nearest opp
        values.append(state[25]) # t3 x
        values.append(state[26]) # t3 y
        values.append(state[12]) # t3 goal open angel
        values.append(state[15]) # t3 distance to nearest opp   
        values.append(state[3]) # ball x
        values.append(state[4]) # ball y
        values.append(state[28]) # d1 x
        values.append(state[29]) # d1 y
        values.append(state[31]) # d2 x
        values.append(state[32]) # d2 y
        values.append(state[34]) # d3 x
        values.append(state[35]) # d3 y
        values.append(state[37]) # d4 x
        values.append(state[38]) # d4 y
        values.append(state[40]) # d5 x
        values.append(state[41]) # d5 y
    else: # defense agents, first opponent (unum 1 - goalie), second opponent (unum 2), third opponent (unum 3), fourth opponent (unum 4), fifth opponent (unum 5)
        values.append(state[40])
        values.append(state[41])
        values.append(state[28])
        values.append(state[29])
        values.append(state[31])
        values.append(state[32])
        values.append(state[34])
        values.append(state[35])
        values.append(state[37])
        values.append(state[38])
        values.append(state[3]) # ball x
        values.append(state[4]) # ball y
        values.append(state[19]) # t1 x
        values.append(state[20]) # t1 y
        values.append(state[22]) # t2 x
        values.append(state[23]) # t2 y
        values.append(state[25]) # t3 x
        values.append(state[26]) # t3 y
        values.append(state[0]) # t4 x
        values.append(state[1]) # t4 y
    return values

def get_action(agent, teammates, opponents, locations, uniform_numbers):
    # agent_next_angle = best_shoot_angle_locations(agent.x_pos, agent.y_pos, locations, uniform_numbers)
    if agent.x_pos >= 0.2 and agent.goal_opening_angle > -0.747394386098 and agent.dist_to_goal < 0.136664020547 and agent.proximity_op > -0.9: # and agent_next_angle > 11:
        action = [hfo.SHOOT]
        # if you manage to pass to any teammate chnge this to consider all teammates
    elif teammates[0].uniform_num != -2:
        safe_index, velocity = distance_to_opponent([agent.x_pos,agent.y_pos], teammates, opponents)
        if safe_index == 'PASS':
            action = [hfo.DRIBBLE] # pass to nearest
        else:
            action = [(hfo.KICK_TO, teammates[safe_index].x_pos, teammates[safe_index].y_pos, velocity)]
    else:
        action = [hfo.DRIBBLE]
    return action

def get_action_old_version(agent, teammate, opponents, locations, uniform_numbers):
    if agent.x_pos >= 0.2:
        agent_next_angle = best_shoot_angle_locations(agent.x_pos, agent.y_pos, locations, uniform_numbers)
        if agent.goal_opening_angle > -0.747394386098 and agent.dist_to_goal < 0.136664020547 and agent.proximity_op > -0.9 and agent_next_angle > 11:
            action = [hfo.SHOOT]
        # if you manage to pass to any teammate chnge this to consider all teammates
        elif teammate.x_pos > agent.x_pos and teammate.goal_angle > -0.74739438609 and teammate.proximity_op > -0.9 and teammate.pass_angle != -2 and abs(teammate.pass_angle) > 0:
            action = [(hfo.PASS, teammate.uniform_num)]
        elif agent.proximity_op < -0.9:
            action = [(hfo.PASS, teammate.uniform_num)]
        else:
            action = [hfo.DRIBBLE]
    else:
        if agent.proximity_op < -0.9 and teammate.proximity_op > -0.7:
            action = [(hfo.PASS, teammate.uniform_num)]
        else:
            print('thi sis nbot workiong') # passing when agent is too near
            action = [hfo.DRIBBLE]
    return action

def best_shoot_angle_locations(agent_x_pos, agent_y_pos, locations, uniform_numbers):
    best_angles = []
    player_coord = np.array([agent_x_pos, agent_y_pos])
    goal_limits = [np.array([0.9, -0.2]), np.array([0.9, 0]), np.array([0.9, 0.2])]
    opponents = process_locations_to_opponents(locations, uniform_numbers, False)
    for goal_limit in goal_limits:
        angles = []
        for op_idx in range(0, len(opponents)):
            op_coord = np.array([opponents[op_idx][0], opponents[op_idx][1]])
            angles.append(calc_angle(goalie=op_coord, player=player_coord, point=goal_limit))
        best_angles.append(min(angles))
    return max(best_angles)

def process_locations_to_opponents(locations, uniform_numbers, grid):
    opponents = []
    if grid:
        idxg = uniform_numbers.index(1.0)
        opponents.append(get_gridno(locations[idxg][0], locations[idxg][1]))
        idxd1 = uniform_numbers.index(2.0)
        opponents.append(get_gridno(locations[idxd1][0], locations[idxd1][1]))
        idxd2 = uniform_numbers.index(3.0)
        opponents.append(get_gridno(locations[idxd2][0], locations[idxd2][1]))
        idxd3 = uniform_numbers.index(4.0)
        opponents.append(get_gridno(locations[idxd3][0], locations[idxd3][1]))
        idxd4 = uniform_numbers.index(5.0)
        opponents.append(get_gridno(locations[idxd4][0], locations[idxd4][1]))
    else:
        idxg = uniform_numbers.index(1.0)
        opponents.append([locations[idxg][0], locations[idxg][1]])
        idxd1 = uniform_numbers.index(2.0)
        opponents.append([locations[idxd1][0], locations[idxd1][1]])
        idxd2 = uniform_numbers.index(3.0)
        opponents.append([locations[idxd2][0], locations[idxd2][1]])
        idxd3 = uniform_numbers.index(4.0)
        opponents.append([locations[idxd3][0], locations[idxd3][1]])
        idxd4 = uniform_numbers.index(5.0)
        opponents.append([locations[idxd4][0], locations[idxd4][1]])
    return opponents

def calc_angle(goalie: np.ndarray, player: np.ndarray, point: np.ndarray):
    for array in [goalie, player, point]:
        if array[0] == -2 or array[1] == -2:
            return 0
    a = np.array(goalie)
    b = np.array(player)
    c = np.array(point)

    ba = a - b
    bc = c - b
    
    np.seterr(invalid='ignore')
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)

# Get area of a triangle
def area(x1, y1, x2, y2, x3, y3):
    return abs((x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2)) / 2.0)

# Check if a point is inside a triangle
def is_inside(x1, y1, x2, y2, x3, y3, x, y):
    A = area (x1, y1, x2, y2, x3, y3)
    A1 = area (x, y, x2, y2, x3, y3)
    A2 = area (x1, y1, x, y, x3, y3)
    A3 = area (x1, y1, x2, y2, x, y)
    # Check if sum of A1, A2 and A3 is equal to A
    if(A == A1 + A2 + A3):
        return True
    else:
        return False

# Get angle terms
def get_angle_terms(agent, opponents_inside_goal):
    angle_terms = []
    if agent.goal_opening_angle > -0.745: # good goal ang
        angle_terms = ['holds(agent_angle(30),0).'] # if changed in ASP change this
    if len(opponents_inside_goal) != 0:
        x_min = agent.grid[0]-1 # only get here if >= 13
        x_max = 19
        y_min = agent.grid[1] if agent.grid[1] < 8 else 8 # from goal limit
        y_max = agent.grid[1] if agent.grid[1] > 11 else 11
        # we know where each opponent will be in this step and next step
        # assuming those locations are static and our agent is the only one who can move in some particular region between some min max
        # calculate the angel values and add the max angles to the ASP array at each location
        # for the first set of locations we know whether its possible or not from the goal open angle
        # so this is only needed for the second location set from models => assuming they are correct
        
        opponents = sorted(opponents_inside_goal, key=lambda x: x[1]) # sorted ascen
        opponents.insert(0,[19,8])
        opponents.append([19,11])
        for agent_x in range(x_min, x_max+1): # calculate angle for whole range
            for agent_y in range(y_min, y_max+1):
                agent_cord = np.array([agent_x, agent_y])
                angle_array = []
                for item_nu in range(len(opponents)-1):
                    angle = calc_angle(np.asarray(opponents[item_nu]), agent_cord, np.asarray(opponents[item_nu+1]))
                    if not math.isnan(angle):
                        angle_array.append(angle)
                if len(angle_array) > 1:
                    max_index = angle_array.index(max(angle_array))
                    angle_terms.append('angle('+str(round(max(angle_array)))+','+str(opponents[max_index][0])+','+str(opponents[max_index][1]) \
                                        +','+str(agent_x)+','+str(agent_y)+').')
    else:
        angle_terms = [] # what if there exists no opponents inside goal no defenders infront / 
    return angle_terms
def get_next_terms(x_min, x_max, y_min, y_max):
    next_txt_file = open("ASP/next_to_terms.txt", "r")
    file_content = next_txt_file.read()
    next_relations = file_content.split("\n")
    next_txt_file.close()
    next_terms = []
    for element in next_relations:
        index_list = list(map(int,re.findall(r'\(.*?\)',element)[0].strip('()').split(',')))
        if index_list[0] >= x_min and index_list[0] <= x_max:
                if index_list[1] >= y_min and index_list[1] <= y_max:
                    if index_list[2] >= x_min and index_list[2] <= x_max:
                        if index_list[3] >= y_min and index_list[3] <= y_max:
                            next_terms.append(element)
    return next_terms
    
def get_limits(agent, teammate, opponents, locations, uniform_numbers):
    opponents_inside_goal = []
    idx = uniform_numbers.index(teammate.uniform_num)
    grid_nearest_teammate = get_gridno(locations[idx][0], locations[idx][1])
    opponents_next_loc = process_locations_to_opponents(locations, uniform_numbers, True)
    # only need to consider the opponents within reach
    # for oppo in opponents:
    #     if oppo.grid[0] >= agent.grid[0]: # oppo infront of agent
    #         if is_inside(19, 11, 19, 8, agent.grid[0], agent.grid[1], oppo.grid[0], oppo.grid[1]):
    #             opponents_inside_goal.append([oppo.grid[0], oppo.grid[1]])
    for oppo_n in opponents_next_loc: # consider only the future step
        if oppo_n[0] >= agent.grid[0]:
            if is_inside(19, 11, 19, 8, agent.grid[0], agent.grid[1], oppo_n[0], oppo_n[1]):
                opponents_inside_goal.append([oppo_n[0], oppo_n[1]])
    new_data = [list(y) for y in set([tuple(x) for x in opponents_inside_goal])] # remove duplicat
    if agent.grid[0] >= 13: # can sho
        x_min = agent.grid[0]-1
        x_max = 19
        y_min = agent.grid[1]-1 if agent.grid[1] >= 1 else agent.grid[1]
        y_max = agent.grid[1]+1 if agent.grid[1] <= 18 else agent.grid[1]
    else:
        x_min = agent.grid[0]
        x_max = 13
        y_min = agent.grid[1]-1 if agent.grid[1] >= 1 else agent.grid[1]
        y_max = agent.grid[1]+1 if agent.grid[1] <= 18 else agent.grid[1]
    x_value, y_value = get_const_xy(agent.grid[0], x_min, x_max, y_min, y_max, teammate, opponents, grid_nearest_teammate, opponents_next_loc)
    prefix = ['#const n = 8.','sorts', x_value, y_value]
    return new_data, x_min, x_max, y_min, y_max, prefix

def get_const_xy(agent_x, x_min, x_max, y_min, y_max, teammate, opponents, grid_nearest_teammate, opponents_next_loc):
    x_min, x_max, y_min, y_max = get_minmax_values(x_min, x_max, y_min, y_max, teammate.grid)
    x_min, x_max, y_min, y_max = get_minmax_values(x_min, x_max, y_min, y_max, grid_nearest_teammate)
    if agent_x >= 13: # can shoot
        for oppo in opponents:
            x_min, x_max, y_min, y_max = get_minmax_values(x_min, x_max, y_min, y_max, oppo.grid)
        for oppo_n in opponents_next_loc:    
            x_min, x_max, y_min, y_max = get_minmax_values(x_min, x_max, y_min, y_max, oppo_n)
        if y_min > 8: # after considering agent_y +-one
            y_min = 8
        if y_max < 11:
            y_max = 11
    else:
        for oppo in opponents:
            if oppo.grid[0] < 13:
                x_min, x_max, y_min, y_max = get_minmax_values(x_min, x_max, y_min, y_max, oppo.grid)
        for oppo_n in opponents_next_loc:
            if oppo_n[0] < 13:
                x_min, x_max, y_min, y_max = get_minmax_values(x_min, x_max, y_min, y_max, oppo_n)

    # x_min = x_min-1 if x_min >= 1 else x_min # to be in safe side => this will make agent values +-two
    # x_max = x_max+1 if x_max <=18 else x_max
    # y_min = y_min-1 if y_min >= 1 else y_min
    # y_max = y_max+1 if y_max <=18 else y_max
    
    x_value = '#x_value = ' + str(x_min)+'..'+str(x_max) + '.'
    y_value = '#y_value = ' + str(y_min)+'..'+str(y_max) + '.'
    return x_value, y_value

def get_minmax_values(x_min, x_max, y_min, y_max, grid):
    if x_min == None or x_min > grid[0]:
        x_min = grid[0]
    if x_max == None or x_max < grid[0]:
        x_max = grid[0]
    if y_min == None or y_min > grid[1]:
        y_min = grid[1]
    if y_max == None or y_max < grid[1]:
        y_max = grid[1]
    return x_min, x_max, y_min, y_max

# Get the grid number of the location
def get_gridno(x,y):
    if x == -2 or y == -2:
        zone =  np.array([-2,-2])
        return zone
        
    if (x >= 0 and x <= 1):
        if (y >= 0 and y <= 1):
            if (x >= 0 and x < 0.1):
                if (y >= 0 and y < 0.1):
                    zone =  np.array([10,9])
                elif (y >= 0.1 and y < 0.2):
                    zone =  np.array([10,8])
                elif (y >= 0.2 and y < 0.3):
                    zone =  np.array([10,7])
                elif (y >= 0.3 and y < 0.4):
                    zone =  np.array([10,6])
                elif (y >= 0.4 and y < 0.5):
                    zone =  np.array([10,5])
                elif (y >= 0.5 and y < 0.6):
                    zone =  np.array([10,4])
                elif (y >= 0.6 and y < 0.7):
                    zone =  np.array([10,3])
                elif (y >= 0.7 and y < 0.8):
                    zone =  np.array([10,2])
                elif (y >= 0.8 and y < 0.9):
                    zone =  np.array([10,1])
                else:
                    zone =  np.array([10,0])
            elif (x >= 0.1 and x < 0.2):
                if (y >= 0 and y < 0.1):
                    zone =  np.array([11,9])
                elif (y >= 0.1 and y < 0.2):
                    zone =  np.array([11,8])
                elif (y >= 0.2 and y < 0.3):
                    zone =  np.array([11,7])
                elif (y >= 0.3 and y < 0.4):
                    zone =  np.array([11,6])
                elif (y >= 0.4 and y < 0.5):
                    zone =  np.array([11,5])
                elif (y >= 0.5 and y < 0.6):
                    zone =  np.array([11,4])
                elif (y >= 0.6 and y < 0.7):
                    zone =  np.array([11,3])
                elif (y >= 0.7 and y < 0.8):
                    zone =  np.array([11,2])
                elif (y >= 0.8 and y < 0.9):
                    zone =  np.array([11,1])
                else:
                    zone =  np.array([11,0])
            elif (x >= 0.2 and x < 0.3):
                if (y >= 0 and y < 0.1):
                    zone =  np.array([12,9])
                elif (y >= 0.1 and y < 0.2):
                    zone =  np.array([12,8])
                elif (y >= 0.2 and y < 0.3):
                    zone =  np.array([12,7])
                elif (y >= 0.3 and y < 0.4):
                    zone =  np.array([12,6])
                elif (y >= 0.4 and y < 0.5):
                    zone =  np.array([12,5])
                elif (y >= 0.5 and y < 0.6):
                    zone =  np.array([12,4])
                elif (y >= 0.6 and y < 0.7):
                    zone =  np.array([12,3])
                elif (y >= 0.7 and y < 0.8):
                    zone =  np.array([12,2])
                elif (y >= 0.8 and y < 0.9):
                    zone =  np.array([12,1])
                else:
                    zone =  np.array([12,0])
            elif (x >= 0.3 and x < 0.4):
                if (y >= 0 and y < 0.1):
                    zone =  np.array([13,9])
                elif (y >= 0.1 and y < 0.2):
                    zone =  np.array([13,8])
                elif (y >= 0.2 and y < 0.3):
                    zone =  np.array([13,7])
                elif (y >= 0.3 and y < 0.4):
                    zone =  np.array([13,6])
                elif (y >= 0.4 and y < 0.5):
                    zone =  np.array([13,5])
                elif (y >= 0.5 and y < 0.6):
                    zone =  np.array([13,4])
                elif (y >= 0.6 and y < 0.7):
                    zone =  np.array([13,3])
                elif (y >= 0.7 and y < 0.8):
                    zone =  np.array([13,2])
                elif (y >= 0.8 and y < 0.9):
                    zone =  np.array([13,1])
                else:
                    zone =  np.array([13,0])
            elif (x >= 0.4 and x < 0.5):
                if (y >= 0 and y < 0.1):
                    zone =  np.array([14,9])
                elif (y >= 0.1 and y < 0.2):
                    zone =  np.array([14,8])
                elif (y >= 0.2 and y < 0.3):
                    zone =  np.array([14,7])
                elif (y >= 0.3 and y < 0.4):
                    zone =  np.array([14,6])
                elif (y >= 0.4 and y < 0.5):
                    zone =  np.array([14,5])
                elif (y >= 0.5 and y < 0.6):
                    zone =  np.array([14,4])
                elif (y >= 0.6 and y < 0.7):
                    zone =  np.array([14,3])
                elif (y >= 0.7 and y < 0.8):
                    zone =  np.array([14,2])
                elif (y >= 0.8 and y < 0.9):
                    zone =  np.array([14,1])
                else:
                    zone =  np.array([14,0])
            elif (x >= 0.5 and x < 0.6):
                if (y >= 0 and y < 0.1):
                    zone =  np.array([15,9])
                elif (y >= 0.1 and y < 0.2):
                    zone =  np.array([15,8])
                elif (y >= 0.2 and y < 0.3):
                    zone =  np.array([15,7])
                elif (y >= 0.3 and y < 0.4):
                    zone =  np.array([15,6])
                elif (y >= 0.4 and y < 0.5):
                    zone =  np.array([15,5])
                elif (y >= 0.5 and y < 0.6):
                    zone =  np.array([15,4])
                elif (y >= 0.6 and y < 0.7):
                    zone =  np.array([15,3])
                elif (y >= 0.7 and y < 0.8):
                    zone =  np.array([15,2])
                elif (y >= 0.8 and y < 0.9):
                    zone =  np.array([15,1])
                else:
                    zone =  np.array([15,0])
            elif (x >= 0.6 and x < 0.7):
                if (y >= 0 and y < 0.1):
                    zone =  np.array([16,9])
                elif (y >= 0.1 and y < 0.2):
                    zone =  np.array([16,8])
                elif (y >= 0.2 and y < 0.3):
                    zone =  np.array([16,7])
                elif (y >= 0.3 and y < 0.4):
                    zone =  np.array([16,6])
                elif (y >= 0.4 and y < 0.5):
                    zone =  np.array([16,5])
                elif (y >= 0.5 and y < 0.6):
                    zone =  np.array([16,4])
                elif (y >= 0.6 and y < 0.7):
                    zone =  np.array([16,3])
                elif (y >= 0.7 and y < 0.8):
                    zone =  np.array([16,2])
                elif (y >= 0.8 and y < 0.9):
                    zone =  np.array([16,1])
                else:
                    zone =  np.array([16,0])
            elif (x >= 0.7 and x < 0.8):
                if (y >= 0 and y < 0.1):
                    zone =  np.array([17,9])
                elif (y >= 0.1 and y < 0.2):
                    zone =  np.array([17,8])
                elif (y >= 0.2 and y < 0.3):
                    zone =  np.array([17,7])
                elif (y >= 0.3 and y < 0.4):
                    zone =  np.array([17,6])
                elif (y >= 0.4 and y < 0.5):
                    zone =  np.array([17,5])
                elif (y >= 0.5 and y < 0.6):
                    zone =  np.array([17,4])
                elif (y >= 0.6 and y < 0.7):
                    zone =  np.array([17,3])
                elif (y >= 0.7 and y < 0.8):
                    zone =  np.array([17,2])
                elif (y >= 0.8 and y < 0.9):
                    zone =  np.array([17,1])
                else:
                    zone =  np.array([17,0])
            elif (x >= 0.8 and x < 0.9):
                if (y >= 0 and y < 0.1):
                    zone =  np.array([18,9])
                elif (y >= 0.1 and y < 0.2):
                    zone =  np.array([18,8])
                elif (y >= 0.2 and y < 0.3):
                    zone =  np.array([18,7])
                elif (y >= 0.3 and y < 0.4):
                    zone =  np.array([18,6])
                elif (y >= 0.4 and y < 0.5):
                    zone =  np.array([18,5])
                elif (y >= 0.5 and y < 0.6):
                    zone =  np.array([18,4])
                elif (y >= 0.6 and y < 0.7):
                    zone =  np.array([18,3])
                elif (y >= 0.7 and y < 0.8):
                    zone =  np.array([18,2])
                elif (y >= 0.8 and y < 0.9):
                    zone =  np.array([18,1])
                else:
                    zone =  np.array([18,0])
            else:
                if (y >= 0 and y < 0.1):
                    zone =  np.array([19,9])
                elif (y >= 0.1 and y < 0.2):
                    zone =  np.array([19,8])
                elif (y >= 0.2 and y < 0.3):
                    zone =  np.array([19,7])
                elif (y >= 0.3 and y < 0.4):
                    zone =  np.array([19,6])
                elif (y >= 0.4 and y < 0.5):
                    zone =  np.array([19,5])
                elif (y >= 0.5 and y < 0.6):
                    zone =  np.array([19,4])
                elif (y >= 0.6 and y < 0.7):
                    zone =  np.array([19,3])
                elif (y >= 0.7 and y < 0.8):
                    zone =  np.array([19,2])
                elif (y >= 0.8 and y < 0.9):
                    zone =  np.array([19,1])
                else:
                    zone =  np.array([19,0])
        else:
            if (x >= 0 and x < 0.1):
                if (y <= 0 and y > -0.1):
                    zone =  np.array([10,10])
                elif (y <= -0.1 and y > -0.2):
                    zone =  np.array([10,11])
                elif (y <= -0.2 and y > -0.3):
                    zone =  np.array([10,12])
                elif (y <= -0.3 and y > -0.4):
                    zone =  np.array([10,13])
                elif (y <= -0.4 and y > -0.5):
                    zone =  np.array([10,14])
                elif (y <= -0.5 and y > -0.6):
                    zone =  np.array([10,15])
                elif (y <= -0.6 and y > -0.7):
                    zone =  np.array([10,16])
                elif (y <= -0.7 and y > -0.8):
                    zone =  np.array([10,17])
                elif (y <= -0.8 and y > -0.9):
                    zone =  np.array([10,18])
                else:
                    zone =  np.array([10,19])
            elif (x >= 0.1 and x < 0.2):
                if (y <= 0 and y > -0.1):
                    zone =  np.array([11,10])
                elif (y <= -0.1 and y > -0.2):
                    zone =  np.array([11,11])
                elif (y <= -0.2 and y > -0.3):
                    zone =  np.array([11,12])
                elif (y <= -0.3 and y > -0.4):
                    zone =  np.array([11,13])
                elif (y <= -0.4 and y > -0.5):
                    zone =  np.array([11,14])
                elif (y <= -0.5 and y > -0.6):
                    zone =  np.array([11,15])
                elif (y <= -0.6 and y > -0.7):
                    zone =  np.array([11,16])
                elif (y <= -0.7 and y > -0.8):
                    zone =  np.array([11,17])
                elif (y <= -0.8 and y > -0.9):
                    zone =  np.array([11,18])
                else:
                    zone =  np.array([11,19])  
            elif (x >= 0.2 and x < 0.3):
                if (y <= 0 and y > -0.1):
                    zone =  np.array([12,10])
                elif (y <= -0.1 and y > -0.2):
                    zone =  np.array([12,11])
                elif (y <= -0.2 and y > -0.3):
                    zone =  np.array([12,12])
                elif (y <= -0.3 and y > -0.4):
                    zone =  np.array([12,13])
                elif (y <= -0.4 and y > -0.5):
                    zone =  np.array([12,14])
                elif (y <= -0.5 and y > -0.6):
                    zone =  np.array([12,15])
                elif (y <= -0.6 and y > -0.7):
                    zone =  np.array([12,16])
                elif (y <= -0.7 and y > -0.8):
                    zone =  np.array([12,17])
                elif (y <= -0.8 and y > -0.9):
                    zone =  np.array([12,18])
                else:
                    zone =  np.array([12,19])  
            elif (x >= 0.3 and x < 0.4):
                if (y <= 0 and y > -0.1):
                    zone =  np.array([13,10])
                elif (y <= -0.1 and y > -0.2):
                    zone =  np.array([13,11])
                elif (y <= -0.2 and y > -0.3):
                    zone =  np.array([13,12])
                elif (y <= -0.3 and y > -0.4):
                    zone =  np.array([13,13])
                elif (y <= -0.4 and y > -0.5):
                    zone =  np.array([13,14])
                elif (y <= -0.5 and y > -0.6):
                    zone =  np.array([13,15])
                elif (y <= -0.6 and y > -0.7):
                    zone =  np.array([13,16])
                elif (y <= -0.7 and y > -0.8):
                    zone =  np.array([13,17])
                elif (y <= -0.8 and y > -0.9):
                    zone =  np.array([13,18])
                else:
                    zone =  np.array([13,19])
            elif (x >= 0.4 and x < 0.5):
                if (y <= 0 and y > -0.1):
                    zone =  np.array([14,10])
                elif (y <= -0.1 and y > -0.2):
                    zone =  np.array([14,11])
                elif (y <= -0.2 and y > -0.3):
                    zone =  np.array([14,12])
                elif (y <= -0.3 and y > -0.4):
                    zone =  np.array([14,13])
                elif (y <= -0.4 and y > -0.5):
                    zone =  np.array([14,14])
                elif (y <= -0.5 and y > -0.6):
                    zone =  np.array([14,15])
                elif (y <= -0.6 and y > -0.7):
                    zone =  np.array([14,16])
                elif (y <= -0.7 and y > -0.8):
                    zone =  np.array([14,17])
                elif (y <= -0.8 and y > -0.9):
                    zone =  np.array([14,18])
                else:
                    zone =  np.array([14,19])
            elif (x >= 0.5 and x < 0.6):
                if (y <= 0 and y > -0.1):
                    zone =  np.array([15,10])
                elif (y <= -0.1 and y > -0.2):
                    zone =  np.array([15,11])
                elif (y <= -0.2 and y > -0.3):
                    zone =  np.array([15,12])
                elif (y <= -0.3 and y > -0.4):
                    zone =  np.array([15,13])
                elif (y <= -0.4 and y > -0.5):
                    zone =  np.array([15,14])
                elif (y <= -0.5 and y > -0.6):
                    zone =  np.array([15,15])
                elif (y <= -0.6 and y > -0.7):
                    zone =  np.array([15,16])
                elif (y <= -0.7 and y > -0.8):
                    zone =  np.array([15,17])
                elif (y <= -0.8 and y > -0.9):
                    zone =  np.array([15,18])
                else:
                    zone =  np.array([15,19])
            elif (x >= 0.6 and x < 0.7):
                if (y <= 0 and y > -0.1):
                    zone =  np.array([16,10])
                elif (y <= -0.1 and y > -0.2):
                    zone =  np.array([16,11])
                elif (y <= -0.2 and y > -0.3):
                    zone =  np.array([16,12])
                elif (y <= -0.3 and y > -0.4):
                    zone =  np.array([16,13])
                elif (y <= -0.4 and y > -0.5):
                    zone =  np.array([16,14])
                elif (y <= -0.5 and y > -0.6):
                    zone =  np.array([16,15])
                elif (y <= -0.6 and y > -0.7):
                    zone =  np.array([16,16])
                elif (y <= -0.7 and y > -0.8):
                    zone =  np.array([16,17])
                elif (y <= -0.8 and y > -0.9):
                    zone =  np.array([16,18])
                else:
                    zone =  np.array([16,19])
            elif (x >= 0.7 and x < 0.8):
                if (y <= 0 and y > -0.1):
                    zone =  np.array([17,10])
                elif (y <= -0.1 and y > -0.2):
                    zone =  np.array([17,11])
                elif (y <= -0.2 and y > -0.3):
                    zone =  np.array([17,12])
                elif (y <= -0.3 and y > -0.4):
                    zone =  np.array([17,13])
                elif (y <= -0.4 and y > -0.5):
                    zone =  np.array([17,14])
                elif (y <= -0.5 and y > -0.6):
                    zone =  np.array([17,15])
                elif (y <= -0.6 and y > -0.7):
                    zone =  np.array([17,16])
                elif (y <= -0.7 and y > -0.8):
                    zone =  np.array([17,17])
                elif (y <= -0.8 and y > -0.9):
                    zone =  np.array([17,18])
                else:
                    zone =  np.array([17,19])
            elif (x >= 0.8 and x < 0.9):
                if (y <= 0 and y > -0.1):
                    zone =  np.array([18,10])
                elif (y <= -0.1 and y > -0.2):
                    zone =  np.array([18,11])
                elif (y <= -0.2 and y > -0.3):
                    zone =  np.array([18,12])
                elif (y <= -0.3 and y > -0.4):
                    zone =  np.array([18,13])
                elif (y <= -0.4 and y > -0.5):
                    zone =  np.array([18,14])
                elif (y <= -0.5 and y > -0.6):
                    zone =  np.array([18,15])
                elif (y <= -0.6 and y > -0.7):
                    zone =  np.array([18,16])
                elif (y <= -0.7 and y > -0.8):
                    zone =  np.array([18,17])
                elif (y <= -0.8 and y > -0.9):
                    zone =  np.array([18,18])
                else:
                    zone =  np.array([18,19])
            else:
                if (y <= 0 and y > -0.1):
                    zone =  np.array([19,10])
                elif (y <= -0.1 and y > -0.2):
                    zone =  np.array([19,11])
                elif (y <= -0.2 and y > -0.3):
                    zone =  np.array([19,12])
                elif (y <= -0.3 and y > -0.4):
                    zone =  np.array([19,13])
                elif (y <= -0.4 and y > -0.5):
                    zone =  np.array([19,14])
                elif (y <= -0.5 and y > -0.6):
                    zone =  np.array([19,15])
                elif (y <= -0.6 and y > -0.7):
                    zone =  np.array([19,16])
                elif (y <= -0.7 and y > -0.8):
                    zone =  np.array([19,17])
                elif (y <= -0.8 and y > -0.9):
                    zone =  np.array([19,18])
                else:
                    zone =  np.array([19,19]) 
    else:
        if (y >= 0 and y <= 1):
            if (x <= 0 and x > -0.1):
                if (y >= 0 and y < 0.1):
                    zone = np.array([9,9])
                elif (y >= 0.1 and y < 0.2):
                    zone = np.array([9,8])
                elif (y >= 0.2 and y < 0.3):
                    zone = np.array([9,7])
                elif (y >= 0.3 and y < 0.4):
                    zone = np.array([9,6])
                elif (y >= 0.4 and y < 0.5):
                    zone = np.array([9,5])
                elif (y >= 0.5 and y < 0.6):
                    zone = np.array([9,4])
                elif (y >= 0.6 and y < 0.7):
                    zone = np.array([9,3])
                elif (y >= 0.7 and y < 0.8):
                    zone = np.array([9,2])
                elif (y >= 0.8 and y < 0.9):
                    zone = np.array([9,1])
                else:
                    zone =  np.array([9,0])
            elif (x <= -0.1 and x > -0.2):
                if (y >= 0 and y < 0.1):
                    zone = np.array([8,9])
                elif (y >= 0.1 and y < 0.2):
                    zone = np.array([8,8])
                elif (y >= 0.2 and y < 0.3):
                    zone = np.array([8,7])
                elif (y >= 0.3 and y < 0.4):
                    zone = np.array([8,6])
                elif (y >= 0.4 and y < 0.5):
                    zone = np.array([8,5])
                elif (y >= 0.5 and y < 0.6):
                    zone = np.array([8,4])
                elif (y >= 0.6 and y < 0.7):
                    zone = np.array([8,3])
                elif (y >= 0.7 and y < 0.8):
                    zone = np.array([8,2])
                elif (y >= 0.8 and y < 0.9):
                    zone = np.array([8,1])
                else:
                    zone =  np.array([8,0])
            elif (x <= -0.2 and x > -0.3):
                if (y >= 0 and y < 0.1):
                    zone = np.array([7,9])
                elif (y >= 0.1 and y < 0.2):
                    zone = np.array([7,8])
                elif (y >= 0.2 and y < 0.3):
                    zone = np.array([7,7])
                elif (y >= 0.3 and y < 0.4):
                    zone = np.array([7,6])
                elif (y >= 0.4 and y < 0.5):
                    zone = np.array([7,5])
                elif (y >= 0.5 and y < 0.6):
                    zone = np.array([7,4])
                elif (y >= 0.6 and y < 0.7):
                    zone = np.array([7,3])
                elif (y >= 0.7 and y < 0.8):
                    zone = np.array([7,2])
                elif (y >= 0.8 and y < 0.9):
                    zone = np.array([7,1])
                else:
                    zone =  np.array([7,0])
            elif (x <= -0.3 and x > -0.4):
                if (y >= 0 and y < 0.1):
                    zone = np.array([6,9])
                elif (y >= 0.1 and y < 0.2):
                    zone = np.array([6,8])
                elif (y >= 0.2 and y < 0.3):
                    zone = np.array([6,7])
                elif (y >= 0.3 and y < 0.4):
                    zone = np.array([6,6])
                elif (y >= 0.4 and y < 0.5):
                    zone = np.array([6,5])
                elif (y >= 0.5 and y < 0.6):
                    zone = np.array([6,4])
                elif (y >= 0.6 and y < 0.7):
                    zone = np.array([6,3])
                elif (y >= 0.7 and y < 0.8):
                    zone = np.array([6,2])
                elif (y >= 0.8 and y < 0.9):
                    zone = np.array([6,1])
                else:
                    zone =  np.array([6,0])
            elif (x <= -0.4 and x > -0.5):
                if (y >= 0 and y < 0.1):
                    zone = np.array([5,9])
                elif (y >= 0.1 and y < 0.2):
                    zone = np.array([5,8])
                elif (y >= 0.2 and y < 0.3):
                    zone = np.array([5,7])
                elif (y >= 0.3 and y < 0.4):
                    zone = np.array([5,6])
                elif (y >= 0.4 and y < 0.5):
                    zone = np.array([5,5])
                elif (y >= 0.5 and y < 0.6):
                    zone = np.array([5,4])
                elif (y >= 0.6 and y < 0.7):
                    zone = np.array([5,3])
                elif (y >= 0.7 and y < 0.8):
                    zone = np.array([5,2])
                elif (y >= 0.8 and y < 0.9):
                    zone = np.array([5,1])
                else:
                    zone =  np.array([5,0])
            elif (x <= -0.5 and x > -0.6):
                if (y >= 0 and y < 0.1):
                    zone = np.array([4,9])
                elif (y >= 0.1 and y < 0.2):
                    zone = np.array([4,8])
                elif (y >= 0.2 and y < 0.3):
                    zone = np.array([4,7])
                elif (y >= 0.3 and y < 0.4):
                    zone = np.array([4,6])
                elif (y >= 0.4 and y < 0.5):
                    zone = np.array([4,5])
                elif (y >= 0.5 and y < 0.6):
                    zone = np.array([4,4])
                elif (y >= 0.6 and y < 0.7):
                    zone = np.array([4,3])
                elif (y >= 0.7 and y < 0.8):
                    zone = np.array([4,2])
                elif (y >= 0.8 and y < 0.9):
                    zone = np.array([4,1])
                else:
                    zone =  np.array([4,0])
            elif (x <= -0.6 and x > -0.7):
                if (y >= 0 and y < 0.1):
                    zone = np.array([3,9])
                elif (y >= 0.1 and y < 0.2):
                    zone = np.array([3,8])
                elif (y >= 0.2 and y < 0.3):
                    zone = np.array([3,7])
                elif (y >= 0.3 and y < 0.4):
                    zone = np.array([3,6])
                elif (y >= 0.4 and y < 0.5):
                    zone = np.array([3,5])
                elif (y >= 0.5 and y < 0.6):
                    zone = np.array([3,4])
                elif (y >= 0.6 and y < 0.7):
                    zone = np.array([3,3])
                elif (y >= 0.7 and y < 0.8):
                    zone = np.array([3,2])
                elif (y >= 0.8 and y < 0.9):
                    zone = np.array([3,1])
                else:
                    zone =  np.array([3,0])
            elif (x <= -0.7 and x > -0.8):
                if (y >= 0 and y < 0.1):
                    zone = np.array([2,9])
                elif (y >= 0.1 and y < 0.2):
                    zone = np.array([2,8])
                elif (y >= 0.2 and y < 0.3):
                    zone = np.array([2,7])
                elif (y >= 0.3 and y < 0.4):
                    zone = np.array([2,6])
                elif (y >= 0.4 and y < 0.5):
                    zone = np.array([2,5])
                elif (y >= 0.5 and y < 0.6):
                    zone = np.array([2,4])
                elif (y >= 0.6 and y < 0.7):
                    zone = np.array([2,3])
                elif (y >= 0.7 and y < 0.8):
                    zone = np.array([2,2])
                elif (y >= 0.8 and y < 0.9):
                    zone = np.array([2,1])
                else:
                    zone =  np.array([2,0])
            elif (x <= -0.8 and x > -0.9):
                if (y >= 0 and y < 0.1):
                    zone = np.array([1,9])
                elif (y >= 0.1 and y < 0.2):
                    zone = np.array([1,8])
                elif (y >= 0.2 and y < 0.3):
                    zone = np.array([1,7])
                elif (y >= 0.3 and y < 0.4):
                    zone = np.array([1,6])
                elif (y >= 0.4 and y < 0.5):
                    zone = np.array([1,5])
                elif (y >= 0.5 and y < 0.6):
                    zone = np.array([1,4])
                elif (y >= 0.6 and y < 0.7):
                    zone = np.array([1,3])
                elif (y >= 0.7 and y < 0.8):
                    zone = np.array([1,2])
                elif (y >= 0.8 and y < 0.9):
                    zone = np.array([1,1])
                else:
                    zone =  np.array([1,0])
            else:
                if (y >= 0 and y < 0.1):
                    zone = np.array([0,9])
                elif (y >= 0.1 and y < 0.2):
                    zone = np.array([0,8])
                elif (y >= 0.2 and y < 0.3):
                    zone = np.array([0,7])
                elif (y >= 0.3 and y < 0.4):
                    zone = np.array([0,6])
                elif (y >= 0.4 and y < 0.5):
                    zone = np.array([0,5])
                elif (y >= 0.5 and y < 0.6):
                    zone = np.array([0,4])
                elif (y >= 0.6 and y < 0.7):
                    zone = np.array([0,3])
                elif (y >= 0.7 and y < 0.8):
                    zone = np.array([0,2])
                elif (y >= 0.8 and y < 0.9):
                    zone = np.array([0,1])
                else:
                    zone =  np.array([0,0])
        else:
            if (x <= 0 and x > -0.1):
                if (y <= 0 and y > -0.1):
                    zone = np.array([9,10])
                elif (y <= -0.1 and y > -0.2):
                    zone = np.array([9,11])
                elif (y <= -0.2 and y > -0.3):
                    zone = np.array([9,12])
                elif (y <= -0.3 and y > -0.4):
                    zone = np.array([9,13])
                elif (y <= -0.4 and y > -0.5):
                    zone = np.array([9,14])
                elif (y <= -0.5 and y > -0.6):
                    zone = np.array([9,15])
                elif (y <= -0.6 and y > -0.7):
                    zone = np.array([9,16])
                elif (y <= -0.7 and y > -0.8):
                    zone = np.array([9,17])
                elif (y <= -0.8 and y > -0.9):
                    zone = np.array([9,18])
                else:
                    zone = np.array([9,19])
            elif (x <= -0.1 and x > -0.2):
                if (y <= 0 and y > -0.1):
                    zone = np.array([8,10])
                elif (y <= -0.1 and y > -0.2):
                    zone = np.array([8,11])
                elif (y <= -0.2 and y > -0.3):
                    zone = np.array([8,12])
                elif (y <= -0.3 and y > -0.4):
                    zone = np.array([8,13])
                elif (y <= -0.4 and y > -0.5):
                    zone = np.array([8,14])
                elif (y <= -0.5 and y > -0.6):
                    zone = np.array([8,15])
                elif (y <= -0.6 and y > -0.7):
                    zone = np.array([8,16])
                elif (y <= -0.7 and y > -0.8):
                    zone = np.array([8,17])
                elif (y <= -0.8 and y > -0.9):
                    zone = np.array([8,18])
                else:
                    zone = np.array([8,19])
            elif (x <= -0.2 and x > -0.3):
                if (y <= 0 and y > -0.1):
                    zone = np.array([7,10])
                elif (y <= -0.1 and y > -0.2):
                    zone = np.array([7,11])
                elif (y <= -0.2 and y > -0.3):
                    zone = np.array([7,12])
                elif (y <= -0.3 and y > -0.4):
                    zone = np.array([7,13])
                elif (y <= -0.4 and y > -0.5):
                    zone = np.array([7,14])
                elif (y <= -0.5 and y > -0.6):
                    zone = np.array([7,15])
                elif (y <= -0.6 and y > -0.7):
                    zone = np.array([7,16])
                elif (y <= -0.7 and y > -0.8):
                    zone = np.array([7,17])
                elif (y <= -0.8 and y > -0.9):
                    zone = np.array([7,18])
                else:
                    zone = np.array([7,19])
            elif (x <= -0.3 and x > -0.4):
                if (y <= 0 and y > -0.1):
                    zone = np.array([6,10])
                elif (y <= -0.1 and y > -0.2):
                    zone = np.array([6,11])
                elif (y <= -0.2 and y > -0.3):
                    zone = np.array([6,12])
                elif (y <= -0.3 and y > -0.4):
                    zone = np.array([6,13])
                elif (y <= -0.4 and y > -0.5):
                    zone = np.array([6,14])
                elif (y <= -0.5 and y > -0.6):
                    zone = np.array([6,15])
                elif (y <= -0.6 and y > -0.7):
                    zone = np.array([6,16])
                elif (y <= -0.7 and y > -0.8):
                    zone = np.array([6,17])
                elif (y <= -0.8 and y > -0.9):
                    zone = np.array([6,18])
                else:
                    zone = np.array([6,19])
            elif (x <= -0.4 and x > -0.5):
                if (y <= 0 and y > -0.1):
                    zone = np.array([5,10])
                elif (y <= -0.1 and y > -0.2):
                    zone = np.array([5,11])
                elif (y <= -0.2 and y > -0.3):
                    zone = np.array([5,12])
                elif (y <= -0.3 and y > -0.4):
                    zone = np.array([5,13])
                elif (y <= -0.4 and y > -0.5):
                    zone = np.array([5,14])
                elif (y <= -0.5 and y > -0.6):
                    zone = np.array([5,15])
                elif (y <= -0.6 and y > -0.7):
                    zone = np.array([5,16])
                elif (y <= -0.7 and y > -0.8):
                    zone = np.array([5,17])
                elif (y <= -0.8 and y > -0.9):
                    zone = np.array([5,18])
                else:
                    zone = np.array([5,19])
            elif (x <= -0.5 and x > -0.6):
                if (y <= 0 and y > -0.1):
                    zone = np.array([4,10])
                elif (y <= -0.1 and y > -0.2):
                    zone = np.array([4,11])
                elif (y <= -0.2 and y > -0.3):
                    zone = np.array([4,12])
                elif (y <= -0.3 and y > -0.4):
                    zone = np.array([4,13])
                elif (y <= -0.4 and y > -0.5):
                    zone = np.array([4,14])
                elif (y <= -0.5 and y > -0.6):
                    zone = np.array([4,15])
                elif (y <= -0.6 and y > -0.7):
                    zone = np.array([4,16])
                elif (y <= -0.7 and y > -0.8):
                    zone = np.array([4,17])
                elif (y <= -0.8 and y > -0.9):
                    zone = np.array([4,18])
                else:
                    zone = np.array([4,19])
            elif (x <= -0.6 and x > -0.7):
                if (y <= 0 and y > -0.1):
                    zone = np.array([3,10])
                elif (y <= -0.1 and y > -0.2):
                    zone = np.array([3,11])
                elif (y <= -0.2 and y > -0.3):
                    zone = np.array([3,12])
                elif (y <= -0.3 and y > -0.4):
                    zone = np.array([3,13])
                elif (y <= -0.4 and y > -0.5):
                    zone = np.array([3,14])
                elif (y <= -0.5 and y > -0.6):
                    zone = np.array([3,15])
                elif (y <= -0.6 and y > -0.7):
                    zone = np.array([3,16])
                elif (y <= -0.7 and y > -0.8):
                    zone = np.array([3,17])
                elif (y <= -0.8 and y > -0.9):
                    zone = np.array([3,18])
                else:
                    zone = np.array([3,19])
            elif (x <= -0.7 and x > -0.8):
                if (y <= 0 and y > -0.1):
                    zone = np.array([2,10])
                elif (y <= -0.1 and y > -0.2):
                    zone = np.array([2,11])
                elif (y <= -0.2 and y > -0.3):
                    zone = np.array([2,12])
                elif (y <= -0.3 and y > -0.4):
                    zone = np.array([2,13])
                elif (y <= -0.4 and y > -0.5):
                    zone = np.array([2,14])
                elif (y <= -0.5 and y > -0.6):
                    zone = np.array([2,15])
                elif (y <= -0.6 and y > -0.7):
                    zone = np.array([2,16])
                elif (y <= -0.7 and y > -0.8):
                    zone = np.array([2,17])
                elif (y <= -0.8 and y > -0.9):
                    zone = np.array([2,18])
                else:
                    zone = np.array([2,19])
            elif (x <= -0.8 and x > -0.9):
                if (y <= 0 and y > -0.1):
                    zone = np.array([1,10])
                elif (y <= -0.1 and y > -0.2):
                    zone = np.array([1,11])
                elif (y <= -0.2 and y > -0.3):
                    zone = np.array([1,12])
                elif (y <= -0.3 and y > -0.4):
                    zone = np.array([1,13])
                elif (y <= -0.4 and y > -0.5):
                    zone = np.array([1,14])
                elif (y <= -0.5 and y > -0.6):
                    zone = np.array([1,15])
                elif (y <= -0.6 and y > -0.7):
                    zone = np.array([1,16])
                elif (y <= -0.7 and y > -0.8):
                    zone = np.array([1,17])
                elif (y <= -0.8 and y > -0.9):
                    zone = np.array([1,18])
                else:
                    zone = np.array([1,19])
            else:
                if (y <= 0 and y > -0.1):
                    zone = np.array([0,10])
                elif (y <= -0.1 and y > -0.2):
                    zone = np.array([0,11])
                elif (y <= -0.2 and y > -0.3):
                    zone = np.array([0,12])
                elif (y <= -0.3 and y > -0.4):
                    zone = np.array([0,13])
                elif (y <= -0.4 and y > -0.5):
                    zone = np.array([0,14])
                elif (y <= -0.5 and y > -0.6):
                    zone = np.array([0,15])
                elif (y <= -0.6 and y > -0.7):
                    zone = np.array([0,16])
                elif (y <= -0.7 and y > -0.8):
                    zone = np.array([0,17])
                elif (y <= -0.8 and y > -0.9):
                    zone = np.array([0,18])
                else:
                    zone = np.array([0,19])
    return zone

def get_has_ball(state, teammate_pos):
    last_dist = 100
    ball_pos = np.array([state[3], state[4]])
    for i in range(len(teammate_pos)):
        dist = np.linalg.norm(teammate_pos[i] - ball_pos)
        if dist <= last_dist:
            has_ball = i
            last_dist = dist
    return has_ball

# get the angle is suitable for shooting or not
def get_angle(ang_value):
    if ang_value > -0.74739:
        angle = 3
    else:
        angle = 0
    return angle

def update_models(data, agent_type):
    new_state = data[-1]
    prev_state = data[-2]
    old_state = data[-3]
    if agent_type == 'offense': # let s write this for the first teammate=> if this is written in this order should the process data written in this order too?
        old_position = [old_state[19], old_state[20]]
        prev_position = [prev_state[19], prev_state[20]]
        new_position = [new_state[19], new_state[20]]
        new_ball_pos = np.array([new_state[3], new_state[4]])
        prev_ball_pos = np.array([prev_state[3], prev_state[4]])
        old_ball_pos = np.array([old_state[3], old_state[4]])
        new_dist_agent_ball = np.linalg.norm(new_position - new_ball_pos)
        prev_dist_agent_ball = np.linalg.norm(prev_position - prev_ball_pos)
        old_dist_agent_ball = np.linalg.norm(old_position - old_ball_pos)
        if int(prev_state[5]) == 1: # ad hoc agent has the ball, others can move or move_to
            dist_old_positive = np.linalg.norm(old_position - np.array([0.4,0.3]))
            dist_old_negative = np.linalg.norm(old_position - np.array([0.4,-0.3]))
            dist_prev_positive = np.linalg.norm(prev_position - np.array([0.4,0.3]))
            dist_prev_negative = np.linalg.norm(prev_position - np.array([0.4,-0.3]))
            dist_new_positive = np.linalg.norm(new_position - np.array([0.4,0.3]))
            dist_new_negative = np.linalg.norm(new_position - np.array([0.4,-0.3]))
            if new_dist_agent_ball <= prev_dist_agent_ball:
                action = '8'
            elif prev_state[1] < 0 and (dist_new_positive <= dist_prev_positive and dist_prev_positive <= dist_old_positive): # can do this as well if any(y < 0 for y in teammate_y):
                action = '(5, 0.4, 0.3)'
            elif dist_new_negative <= dist_prev_negative and dist_prev_negative <= dist_old_negative:
                action = '(5, 0.4, -0.3)'
            else:
                action = '8'
        else: # teamamte has the ball? or the ball is in the middle of a pass?
            if new_dist_agent_ball < 0.1: # agent has the ball
                if prev_dist_agent_ball < 0.1 or old_dist_agent_ball < 0.1: # agent already had the ball
                    action = '11'
                else: # agent now has the ball - action had been move towards the ball prev
                    action = '8'
            else: # agent does not has the ball
                if prev_dist_agent_ball < 0.1 or old_dist_agent_ball < 0.1: # prev it had the ball - possble action will be pass or a shoot
                    action = '(10, 7)'
                else:
                    # agent does nott has the ball - prev either it did not => it is moving towards the ball?
                    action = '8' # this should be changed later => the agent may move towards the ball only when another agentt is not near to it than itself
    elif agent_type == 'defense':
        # defense agent => possible actions 8,16,15
        action = do_defense_action(4, 4, old_state, prev_state, new_state)
    elif agent_type == 'goalie':
        # goalie agent => possible actions Actions (5, -0.75, 0.175), (5, -0.65, 0), (7,), (5, -0.75, -0.175)
        action = get_goalie_action(old_state, prev_state, new_state)
        # have not changed location => tuen action
    return action

# defense1 (except goalie)
def do_defense_action(num_opponents, num_teammates, old_state, prev_state, new_state):
    if new_state[30] == 2:
        new_position = [new_state[28], new_state[29]]
    elif new_state[33] == 2:
        new_position = [new_state[31], new_state[32]]
    elif new_state[36] == 2:
        new_position = [new_state[34], new_state[35]]
    elif new_state[39] == 2:
        new_position = [new_state[37], new_state[38]]
    else:
        new_position = [new_state[40], new_state[41]]

    if prev_state[30] == 2:
        prev_position = [prev_state[28], prev_state[29]]
    elif prev_state[33] == 2:
        prev_position = [prev_state[31], prev_state[32]]
    elif prev_state[36] == 2:
        prev_position = [prev_state[34], prev_state[35]]
    elif prev_state[39] == 2:
        prev_position = [prev_state[37], prev_state[38]]
    else:
        prev_position = [prev_state[40], prev_state[41]]

    new_ball_pos = np.array([new_state[3], new_state[4]])
    prev_ball_pos = np.array([prev_state[3], prev_state[4]])

    ball_toward_goal = ball_moving_toward_goal(new_ball_pos[0], new_ball_pos[1], prev_ball_pos[0], prev_ball_pos[1])
    ball_nearer_goal = ball_nearer_to_goal(new_ball_pos[0], new_ball_pos[1], new_position[0], new_position[1]) # than agent
    new_ball_sorted_list = get_sorted_opponents(new_state, num_opponents, num_teammates, pos_x=new_ball_pos[0], pos_y=new_ball_pos[1]) # get offense sorted towards ball

    new_dist_to_goal = get_dist_normalized(new_position[0], new_position[1], 0.9, 0.0)
    prev_dist_to_goal = get_dist_normalized(prev_position[0], prev_position[1], 0.9, 0.0)

    ball_free = new_ball_sorted_list[0][0] >= 0.1
    if (not ball_free) and math.isclose(new_dist_to_goal, prev_dist_to_goal, rel_tol=0.001):
        action = hfo.REORIENT
    elif (not ball_free) and ball_toward_goal and new_dist_to_goal < prev_dist_to_goal and (not ball_nearer_goal):
        action = (hfo.MARK_PLAYER, new_ball_sorted_list[0][3]) # this is almost same as REDUCE_ANGLE_TO_GOAL
    else:
        action = hfo.MOVE
    return action

def get_goalie_action(old_state, prev_state, new_state):
    if new_state[30] == 1:
        new_position = np.array([new_state[28], new_state[29]])
    elif new_state[33] == 1:
        new_position = np.array([new_state[31], new_state[32]])
    elif new_state[36] == 1:
        new_position = np.array([new_state[34], new_state[35]])
    elif new_state[39] == 1:
        new_position = np.array([new_state[37], new_state[38]])
    else:
        new_position = np.array([new_state[40], new_state[41]])

    if prev_state[30] == 1:
        prev_position = np.array([prev_state[28], prev_state[29]])
    elif prev_state[33] == 1:
        prev_position = np.array([prev_state[31], prev_state[32]])
    elif prev_state[36] == 1:
        prev_position = np.array([prev_state[34], prev_state[35]])
    elif prev_state[39] == 1:
        prev_position = np.array([prev_state[37], prev_state[38]])
    else:
        prev_position = np.array([prev_state[40], prev_state[41]])

    new_dist_to_minus_minus = np.linalg.norm(new_position - np.array([-0.75, -0.175]))
    prev_dist_to_minus_minus = np.linalg.norm(prev_position - np.array([-0.75, -0.175]))
    new_dist_to_minus_positive = np.linalg.norm(new_position - np.array([-0.75, 0.175]))
    prev_dist_to_minus_positive = np.linalg.norm(prev_position - np.array([-0.75, 0.175]))

    if math.isclose(new_dist_to_minus_minus, prev_dist_to_minus_minus, rel_tol=0.001) and math.isclose(new_dist_to_minus_positive, prev_dist_to_minus_positive, rel_tol=0.001):
        action = hfo.REORIENT
    elif new_dist_to_minus_minus < prev_dist_to_minus_minus:
        action = (5, -0.75, -0.175)
    elif new_dist_to_minus_positive < prev_dist_to_minus_positive:
        action = (5, -0.75, 0.175)
    else:
        action = 7
    return action

def ball_moving_toward_goal(ball_pos_x, ball_pos_y, old_ball_pos_x, old_ball_pos_y):
    return (get_dist_normalized(ball_pos_x, ball_pos_y, 0.9, 0.0) < min(1.504052352, get_dist_normalized(old_ball_pos_x, old_ball_pos_y, 0.9, 0.0)))

def ball_nearer_to_goal(ball_pos_x, ball_pos_y, agent_pos_x, agent_pos_y):
    return (get_dist_normalized(ball_pos_x, ball_pos_y, 0.9, 0.0) < min(1.504052352, get_dist_normalized(agent_pos_x, agent_pos_y, 0.9, 0.0)))
