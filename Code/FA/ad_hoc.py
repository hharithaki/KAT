import subprocess
import re
import math
import time
import utils
import numpy as np
from sklweka.classifiers import Classifier
from sklweka.dataset import Instance, missing_value

asp_learner = 'ASP/learner.sp'
display_marker = 'display'
guard1_model_file = 'models/guard1.model'
guard2_model_file = 'models/guard2.model'
guard3_model_file = 'models/guard3.model'
atttacker1_model_file = 'models/attacker1.model'
atttacker2_model_file = 'models/attacker2.model'
atttacker3_model_file = 'models/attacker3.model'
num_guards = 3
num_attackers = 3
const_term = ['#const n = 13.']
next_dir_terms = ['next_dir(north,north_east1).','next_dir(north_east1,north_east2).','next_dir(north_east2,north_east).',
                  'next_dir(north_east,east1).','next_dir(east1,east2).','next_dir(east2,east).',
                  'next_dir(east,south_east1).','next_dir(south_east1,south_east2).','next_dir(south_east2,south_east).',
                  'next_dir(south_east,south1).','next_dir(south1,south2).','next_dir(south2,south).',
                  'next_dir(south,south_west1).','next_dir(south_west1,south_west2).','next_dir(south_west2,south_west).',
                  'next_dir(south_west,west1).','next_dir(west1,west2).','next_dir(west2,west).',
                  'next_dir(west,north_west1).','next_dir(north_west1,north_west2).','next_dir(north_west2,north_west).',
                  'next_dir(north_west,north1).','next_dir(north1,north2).','next_dir(north2,north).']

def learner(obs, alive_ag, previous_actions, step_nu):
    if obs[0][0] == 1.0: # ad hoc agent alive
        preASPLearner_file = 'ASP/learner_pre.sp'
        reader = open(preASPLearner_file, 'r')
        pre_asp = reader.read()
        reader.close()
        pre_asp_split = pre_asp.split('\n')
        display_marker_index = pre_asp_split.index(display_marker)
        input, next_terms, facing_terms, goal_term, goal_guard, learner_grid, learner_dir = get_actions_terms(obs, alive_ag, previous_actions)
        answer_list = runASPLearner(obs, input, next_terms, facing_terms, goal_term, goal_guard, pre_asp_split, display_marker_index)
        actions = process_answerlist(answer_list, learner_dir, learner_grid)
    else:
        actions = [0,0,0]
    return actions

# look ahead 4 timesteps using the models and another instance of the fortattack env
# assuming ad hoc agent does nothing
def get_actions_terms(obs, alive_ag, previous_actions):
    input = []
    for timestep in range(2): # consider done?
        future_actions = predict_class(obs, alive_ag, previous_actions)
        future_actions[0] = 0 # ad ho agent set as do nothing
        test_env = utils.make_test_env(obs)
        for agent in range(6):
            previous_actions[agent][0] = previous_actions[agent][1]
            previous_actions[agent][1] = future_actions[agent]
        input, learner_grid, learner_dir, learner_pos, guards_info, attackers_info, x_min, x_max, y_min, y_max = get_terms(obs, timestep, input)
        obs, done, alive_ag, info = test_env.step(future_actions)
    next_terms, facing_terms, goal_term, goal_guard = get_additional_terms(learner_pos,learner_grid,guards_info,attackers_info,x_min,x_max,y_min,y_max)
    return input, next_terms, facing_terms, goal_term, goal_guard, learner_grid, learner_dir

# set actions in order
def process_answerlist(answer_list, learner_dir, learner_grid):
    actions = [0]
    if len(answer_list) > 1: # removed situations also include situation where no of steps is not enough(empty (NOTE:not{})) or too much(error)
        action_list = []
        answer_split = ((answer_list.rstrip().split('\n\n'))[0]).strip('{}\n').split(', ')
        for element in answer_split:
            for i in range(len(answer_split)):
                if re.search(rf',{i+1}\)$',element) != None:
                    action_list.insert(i, element)
        actions = map_actions(action_list, learner_dir, learner_grid) if len(action_list) > 0 else [0]
    if len(actions) > 3: # change depend on the results
        actions = actions[:3]
    return actions

# map actions with the fort-attack env action numbers
def map_actions(action_list, learner_dir, learner_grid):
    actions = []
    for element in action_list:
        if element.startswith('occurs(shoot('):
            action = 7
        elif element.startswith('occurs(rotate('):
            new_dir = (re.search(r'occurs\(rotate\(learner,(.*)\),', element)).group(1)
            if learner_dir == 'north' and new_dir == 'north_east1':
                action = 6
                learner_dir = 'north_east1'
            elif learner_dir == 'north_east1' and new_dir == 'north_east2':
                action = 6
                learner_dir = 'north_east2'
            elif learner_dir == 'north_east2' and new_dir == 'north_east':
                action = 6
                learner_dir = 'north_east'
            elif learner_dir == 'north' and new_dir == 'north2':
                action = 5
                learner_dir = 'north2'
            elif learner_dir == 'north2' and new_dir == 'north1':
                action = 5
                learner_dir = 'north1'
            elif learner_dir == 'north1' and new_dir == 'north_west':
                action = 5
                learner_dir = 'north_west'
            elif learner_dir == 'north_east' and new_dir == 'east1':
                action = 6
                learner_dir = 'east1'
            elif learner_dir == 'east1' and new_dir == 'east2':
                action = 6
                learner_dir = 'east2'
            elif learner_dir == 'east2' and new_dir == 'east':
                action = 6
                learner_dir = 'east'
            elif learner_dir == 'north_east' and new_dir == 'north_east2':
                action = 5
                learner_dir = 'north_east2'
            elif learner_dir == 'north_east2' and new_dir == 'north_east1':
                action = 5
                learner_dir = 'north_east1'
            elif learner_dir == 'north_east1' and new_dir == 'north':
                action = 5
                learner_dir = 'north'
            elif learner_dir == 'east' and new_dir == 'south_east1':
                action = 6
                learner_dir = 'south_east1'
            elif learner_dir == 'south_east1' and new_dir == 'south_east2':
                action = 6
                learner_dir = 'south_east1'
            elif learner_dir == 'south_east2' and new_dir == 'south_east':
                action = 6
                learner_dir = 'south_east'
            elif learner_dir == 'east' and new_dir == 'east2':
                action = 5
                learner_dir = 'east2'
            elif learner_dir == 'east2' and new_dir == 'east1':
                action = 5
                learner_dir = 'east1'
            elif learner_dir == 'east1' and new_dir == 'north_east':
                action = 5
                learner_dir = 'north_east'
            elif learner_dir == 'south_east' and new_dir == 'south1':
                action = 6
                learner_dir = 'south1'
            elif learner_dir == 'south1' and new_dir == 'south2':
                action = 6
                learner_dir = 'south2'
            elif learner_dir == 'south2' and new_dir == 'south':
                action = 6
                learner_dir = 'south'
            elif learner_dir == 'south_east' and new_dir == 'south_east2':
                action = 5
                learner_dir = 'south_east2'
            elif learner_dir == 'south_east2' and new_dir == 'south_east1':
                action = 5
                learner_dir = 'south_east1'
            elif learner_dir == 'south_east1' and new_dir == 'east':
                action = 5
                learner_dir = 'east'
            elif learner_dir == 'south' and new_dir == 'south_west1':
                action = 6
                learner_dir = 'south_west1'
            elif learner_dir == 'south_west1' and new_dir == 'south_west2':
                action = 6
                learner_dir = 'south_west2'
            elif learner_dir == 'south_west2' and new_dir == 'south_west':
                action = 6
                learner_dir = 'south_west'
            elif learner_dir == 'south' and new_dir == 'south2':
                action = 5
                learner_dir = 'south2'
            elif learner_dir == 'south2' and new_dir == 'south1':
                action = 5
                learner_dir = 'south1'
            elif learner_dir == 'south1' and new_dir == 'south_east':
                action = 5
                learner_dir = 'south_east'
            elif learner_dir == 'south_west' and new_dir == 'west1':
                action = 6
                learner_dir = 'west1'
            elif learner_dir == 'west1' and new_dir == 'west2':
                action = 6
                learner_dir = 'west2'
            elif learner_dir == 'west2' and new_dir == 'west':
                action = 6
                learner_dir = 'west'
            elif learner_dir == 'south_west' and new_dir == 'south_west2':
                action = 5
                learner_dir = 'south_west2'
            elif learner_dir == 'south_west2' and new_dir == 'south_west1':
                action = 5
                learner_dir = 'south_west1'
            elif learner_dir == 'south_west1' and new_dir == 'south':
                action = 5
                learner_dir = 'south'
            elif learner_dir == 'west' and new_dir == 'north_west1':
                action = 6
                learner_dir = 'north_west1'
            elif learner_dir == 'north_west1' and new_dir == 'north_west2':
                action = 6
                learner_dir = 'north_west2'
            elif learner_dir == 'north_west2' and new_dir == 'north_west':
                action = 6
                learner_dir = 'north_west'
            elif learner_dir == 'west' and new_dir == 'west2':
                action = 5
                learner_dir = 'west2'
            elif learner_dir == 'west2' and new_dir == 'west1':
                action = 5
                learner_dir = 'west1'
            elif learner_dir == 'west1' and new_dir == 'south_west':
                action = 5
                learner_dir = 'south_west'
            elif learner_dir == 'north_west' and new_dir == 'north1':
                action = 6
                learner_dir = 'north1'
            elif learner_dir == 'north1' and new_dir == 'north2':
                action = 6
                learner_dir = 'north2'
            elif learner_dir == 'north2' and new_dir == 'north':
                action = 6
                learner_dir = 'north'
            elif learner_dir == 'north_west' and new_dir == 'north_west2':
                action = 5
                learner_dir = 'north_west2'
            elif learner_dir == 'north_west2' and new_dir == 'north_west1':
                action = 5
                learner_dir = 'north_west1'
            elif learner_dir == 'north_west1' and new_dir == 'west':
                action = 5
                learner_dir = 'west'
        elif element.startswith('occurs(move('):
            new_grid = list(map(int,(re.search(r'occurs\(move\(learner,(.*)\),', element)).group(1).split(',')))
            if learner_grid[0] < int(new_grid[0]):
                action = 1
                learner_grid = new_grid
            elif learner_grid[0] > int(new_grid[0]):
                action = 2
                learner_grid = new_grid
            elif learner_grid[1] < int(new_grid[1]):
                action = 3
                learner_grid = new_grid
            elif learner_grid[1] > int(new_grid[1]):
                action = 4
                learner_grid = new_grid
            else:
                action = 0
        actions.append(action)
    return actions

# return answer sets for the new ASP file
def runASPLearner(obs, input, next_terms, facing_terms, goal_term, goal_guard, pre_asp_split, display_marker_index):
    if goal_guard:
        asp_split = const_term + pre_asp_split[:display_marker_index] + next_terms + facing_terms + goal_term + input + pre_asp_split[display_marker_index:]
    else:
        asp_split = const_term + pre_asp_split[:display_marker_index] + next_dir_terms + next_terms + facing_terms + goal_term + input + pre_asp_split[display_marker_index:]
    asp = '\n'.join(asp_split)
    f1 = open(asp_learner, 'w')
    f1.write(asp)
    f1.close()
    start = time.time()
    answer = subprocess.check_output('java -jar ASP/sparc.jar ' +asp_learner+' -A',shell=True)
    end = time.time()
    answer_split = (answer.decode('ascii'))
    # try:
    #     answer = subprocess.check_output('timeout 10 java -jar ASP/sparc.jar ' +asp_learner+' -A',shell=True)
    #     answer_split = (answer.decode('ascii'))
    # except subprocess.CalledProcessError as e:
    #     print('got timeout exception')
    #     answer_split = []
    return answer_split

# process obs and set the fluents that need to be included and considered in the new ASP program
# ALEADY same loop is running THREE times inside this code. NEED TO OPTIMIZE.
def get_terms(obs, step, input):
    x_min = None
    x_max = None
    y_min = None
    y_max = None
    guards_info, attackers_info = ([] for ag in range(2))
    for agent in range(len(obs)):
        grid = get_gridno(obs[agent][1], obs[agent][2])
        x_min, x_max, y_min, y_max = get_minmax_values(x_min, x_max, y_min, y_max, grid)
        direction = get_direction(obs[agent][3])
        if direction == None:
            direction = 'south' if agent < num_guards else 'north'
        if agent == 0: # ad hoc agent
            agent_name = 'learner'
            learner_grid = grid
            learner_dir = direction
            learner_pos = [obs[agent][1],obs[agent][2]]
            input.append('holds(in('+agent_name+','+str(grid[0])+','+str(grid[1])+'),'+str(step)+').')
            input.append('holds(face('+agent_name+','+direction+'),'+str(step)+').')
            input.append('-holds(shot('+agent_name+'),'+str(step)+').')
        else:
            agent_name = 'guard'+str(agent+1) if agent < num_guards else 'attacker'+str(agent-2)
            input.append('holds(agent_in('+agent_name+','+str(grid[0])+','+str(grid[1])+'),'+str(step)+').')
            input.append('holds(agent_face('+agent_name+','+direction+'),'+str(step)+').')
            if obs[agent][0] == 1.0:
                input.append('-holds(shot('+agent_name+'),'+str(step)+').')
            else:
                input.append('holds(shot('+agent_name+'),'+str(step)+').')
            info_list = [agent,obs[agent][0],obs[agent][1],obs[agent][2],grid[0],grid[1]]
            if agent < num_guards:
                guards_info.append(info_list)
            else:
                attackers_info.append(info_list)
    return input,learner_grid,learner_dir,learner_pos,guards_info,attackers_info,x_min,x_max,y_min,y_max

# get goal, next_to and facing terms
def get_additional_terms(learner_pos,learner_grid,guards_info,attackers_info,x_min,x_max,y_min,y_max):
    goal_term,x_min,x_max,y_min,y_max,goal_guard = get_goal(learner_pos,learner_grid,guards_info,attackers_info,x_min,x_max,y_min,y_max)
    next_terms = get_next_terms(x_min,x_max,y_min,y_max)
    if goal_guard:
        facing_terms = []
    else:
        facing_terms = get_facing_terms(x_min,x_max,y_min,y_max)
    return next_terms, facing_terms, goal_term, goal_guard

# decide the goal depending on the observations
def get_goal(learner_pos,learner_grid,guards_info,attackers_info,x_min,x_max,y_min,y_max):
    danger_attackers,updated_guards = ([] for ll in range(2))
    door = np.array([0,0.8])
    learner = np.array(learner_pos)
    goal = ['goal(I) :- -holds(reached_fort(attacker1),I), -holds(reached_fort(attacker2),I), -holds(reached_fort(attacker1),3).']
    goal_guard = True
    for attacker in range(len(attackers_info)):
        if attackers_info[attacker][1] == 1.0 and attackers_info[attacker][5] > 6: # attacker in danger zone
            arr = np.array([float(attackers_info[attacker][2]),float(attackers_info[attacker][3])])
            attackers_info[attacker].append(np.sqrt(np.sum(np.square(arr-learner))))
            attackers_info[attacker].append(np.sqrt(np.sum(np.square(arr-door))))
            danger_attackers.append(attackers_info[attacker])
    if not danger_attackers: # spread
        for guard in range(len(guards_info)):
            grid_dist_x = abs(learner_grid[0]-guards_info[guard][4])
            grid_dist_y = abs(learner_grid[1]-guards_info[guard][5])
            if guards_info[guard][1] == 1.0 and (grid_dist_x <= 6 or grid_dist_y <= 2): # should we change this y too?
                arr = np.array([float(guards_info[guard][2]),float(guards_info[guard][3])])
                guards_info[guard].append(np.sqrt(np.sum(np.square(arr-learner))))
                updated_guards.append(guards_info[guard])
    else:
        # shoot nearest attacker to learner or fort? or both?
        # for now lets shoot the nearest attacker to the learner
        dist_to_learner = [item[6] for item in danger_attackers]
        min_value = min(dist_to_learner)
        min_index = dist_to_learner.index(min_value)
        attacker_index = danger_attackers[min_index][0]
        goal_guard = False
        goal = ['goal(I) :- holds(agent_shot(attacker'+str(attacker_index-2)+'),I).']
        if danger_attackers[min_index][4] >= learner_grid[0]:
            x_max = danger_attackers[min_index][4]
            x_min = learner_grid[0]
        else:
            x_min = danger_attackers[min_index][4]
            x_max = learner_grid[0]
        if danger_attackers[min_index][5] >= learner_grid[1]:
            y_max = danger_attackers[min_index][5]
            y_min = learner_grid[1]
        else:
            y_min = danger_attackers[min_index][5]
            y_max = learner_grid[1]
        return goal,x_min,x_max,y_min,y_max,goal_guard
    if len(updated_guards) != 0:
        dist_to_learner = [item[6] for item in updated_guards]
        min_value = min(dist_to_learner)
        min_index = dist_to_learner.index(min_value)
        guard_index = updated_guards[min_index][0]
        if updated_guards[min_index][4] >= learner_grid[0]:
            x_max = updated_guards[min_index][4]
            x_min = (updated_guards[min_index][4]-8) if (updated_guards[min_index][4]-8) > 0 else 0
        else:
            x_min = updated_guards[min_index][4]
            x_max = (updated_guards[min_index][4]+8) if (updated_guards[min_index][4]+8) < 19 else 19
        if updated_guards[min_index][5] >= learner_grid[1]:
            y_max = updated_guards[min_index][5]
            y_min = (updated_guards[min_index][5]-4) if (updated_guards[min_index][5]-4) > 0 else 0
        else:
            y_min = updated_guards[min_index][5]
            y_max = (updated_guards[min_index][5]+4) if (updated_guards[min_index][5]+4) < 15 else 15
        goal = ['goal(I) :- holds(distance_gur_to_lea(guard'+str(guard_index+1)+',6,Y),I).']
    return goal,x_min,x_max,y_min,y_max,goal_guard

# return the predicted actions for the other agents as a list/array
def predict_class(obs, alive_ag, previous_actions):
    values = process_data(obs, alive_ag, previous_actions)
    predict_actions = []
    for agent in range(len(obs)):
		# lets assume all guiards are policy1 and all the attackers are policy1
    	# load model
        if agent < num_guards:
            model, header = Classifier.deserialize(guard1_model_file)
        else:
            model, header = Classifier.deserialize(atttacker1_model_file)
	    # create new instance
        inst = Instance.create_instance(values[agent])
        inst.dataset = header
        # make prediction
        index = model.classify_instance(inst)
        predict_actions.append(int(header.class_attribute.value(int(index))))
    return predict_actions

def process_data(obs, alive_ag, previous_actions):
    door = np.array([0,0.8])
    dist_to, alive_att, alive_gua, x_cord, y_cord, orient, x_vel, y_vel = ([] for ia in range(8))

    for j in range(len(obs)):
        x_cord.append(obs[j][1])
        y_cord.append(obs[j][2])
        orient.append(obs[j][3])
        x_vel.append(obs[j][4])
        y_vel.append(obs[j][5])

    for i in range(len(obs)):
        arr = np.array([float(obs[i][1]),float(obs[i][2])])
        distToDoor = np.sqrt(np.sum(np.square(arr-door)))
        dist_to.append(distToDoor)  

    if len(alive_ag) == 0:
        alive_att = np.full(len(obs),2)
        # alive_gua = np.full(len(obs),2)
    else:
        for k in range(len(alive_ag)):
            alive_att.append(alive_ag[k][0])
            # alive_gua.append(alive_ag[k][1])
	
    data = zip(x_cord, y_cord, orient, x_vel, y_vel, alive_att, dist_to)
    values = get_features(data, previous_actions)
    return values

def get_features(data, previous_actions):
	data_array = np.array(list(data))
	values = []
	com_values_p1, com_values_p2 = get_common_values(data_array)
	for agent in range(len(data_array)):
		values.append(com_values_p1 + [previous_actions[agent][-1]] + [previous_actions[agent][-2]] + com_values_p2 + [missing_value()])
	return values

def get_next_terms(x_min,x_max,y_min,y_max):
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
    
def get_facing_terms(x_min,x_max,y_min,y_max):
    facing_txt_file = open("ASP/facing_terms.txt", "r")
    file_content = facing_txt_file.read()
    facing_relations = file_content.split("\n")
    facing_txt_file.close()
    facing_terms = []
    for element in facing_relations:
        index_list = list(re.findall(r'\(.*?\)',element)[0].strip('()').split(','))
        index_list.pop(0)
        index_list = list(map(int,index_list))
        if index_list[0] >= x_min and index_list[0] <= x_max:
            if index_list[1] >= y_min and index_list[1] <= y_max:
                if index_list[2] >= x_min and index_list[2] <= x_max:
                    if index_list[3] >= y_min and index_list[3] <= y_max:
                        facing_terms.append(element)
    return facing_terms

def get_common_values(new_obs):
    com_p1,com_p2 = ([] for ia in range(2))
    dist_att_only = new_obs[num_guards:,6]
    sort_att_dist = np.sort(dist_att_only)
    min_val = sort_att_dist[0]
    next_min = sort_att_dist[1]

    com_p1.append(min_val) # distance to nearest attacker from fort
    com_p1.append(next_min) # distance to next nearest attacker from fort 
    com_p1.append(num_attackers - new_obs[0,5]) # num of dead attackers

    for i in range(6):
        # Cartesian coordinates
        x = new_obs[i][0]
        y = new_obs[i][1]
        com_p2.append(x)
        com_p2.append(y)

        # polar coordinates
        r = np.sqrt(np.square(x)+np.square(y)) # from the center
        theta = math.atan2(y,x) # orientation from center
        com_p2.append(r)
        com_p2.append(theta)

        # grid coordinates
        zone = get_gridno(x,y)
        com_p2.append(zone[0])
        com_p2.append(zone[1])

        com_p2.append(new_obs[i][2]) # orientation in radians

        com_p2.append(new_obs[i][3]) # agent velocity x
        com_p2.append(new_obs[i][4]) # agent velocity y

        com_p2.append(new_obs[i][6]) # agent distance fort

    return com_p1, com_p2

# def getHistory(self):
# 	return self.history    

def get_gridno(x,y):
    if (x >= 0 and x <= 1):
        if (y >= 0 and y <= 0.8):
            if (x >= 0 and x < 0.1):
                if (y >= 0 and y < 0.1):
                    zone =  np.array([10,8])
                elif (y >= 0.1 and y < 0.2):
                    zone =  np.array([10,9])
                elif (y >= 0.2 and y < 0.3):
                    zone =  np.array([10,10])
                elif (y >= 0.3 and y < 0.4):
                    zone =  np.array([10,11])
                elif (y >= 0.4 and y < 0.5):
                    zone =  np.array([10,12])
                elif (y >= 0.5 and y < 0.6):
                    zone =  np.array([10,13])
                elif (y >= 0.6 and y < 0.7):
                    zone =  np.array([10,14])
                else:
                    zone =  np.array([10,15])
            elif (x >= 0.1 and x < 0.2):
                if (y >= 0 and y < 0.1):
                    zone =  np.array([11,8])
                elif (y >= 0.1 and y < 0.2):
                    zone =  np.array([11,9])
                elif (y >= 0.2 and y < 0.3):
                    zone =  np.array([11,10])
                elif (y >= 0.3 and y < 0.4):
                    zone =  np.array([11,11])
                elif (y >= 0.4 and y < 0.5):
                    zone =  np.array([11,12])
                elif (y >= 0.5 and y < 0.6):
                    zone =  np.array([11,13])
                elif (y >= 0.6 and y < 0.7):
                    zone =  np.array([11,14])
                else:
                    zone =  np.array([11,15])
            elif (x >= 0.2 and x < 0.3):
                if (y >= 0 and y < 0.1):
                    zone =  np.array([12,8])
                elif (y >= 0.1 and y < 0.2):
                    zone =  np.array([12,9])
                elif (y >= 0.2 and y < 0.3):
                    zone =  np.array([12,10])
                elif (y >= 0.3 and y < 0.4):
                    zone =  np.array([12,11])
                elif (y >= 0.4 and y < 0.5):
                    zone =  np.array([12,12])
                elif (y >= 0.5 and y < 0.6):
                    zone =  np.array([12,13])
                elif (y >= 0.6 and y < 0.7):
                    zone =  np.array([12,14])
                else:
                    zone =  np.array([12,15])
            elif (x >= 0.3 and x < 0.4):
                if (y >= 0 and y < 0.1):
                    zone =  np.array([13,8])
                elif (y >= 0.1 and y < 0.2):
                    zone =  np.array([13,9])
                elif (y >= 0.2 and y < 0.3):
                    zone =  np.array([13,10])
                elif (y >= 0.3 and y < 0.4):
                    zone =  np.array([13,11])
                elif (y >= 0.4 and y < 0.5):
                    zone =  np.array([13,12])
                elif (y >= 0.5 and y < 0.6):
                    zone =  np.array([13,13])
                elif (y >= 0.6 and y < 0.7):
                    zone =  np.array([13,14])
                else:
                    zone =  np.array([13,15])
            elif (x >= 0.4 and x < 0.5):
                if (y >= 0 and y < 0.1):
                    zone =  np.array([14,8])
                elif (y >= 0.1 and y < 0.2):
                    zone =  np.array([14,9])
                elif (y >= 0.2 and y < 0.3):
                    zone =  np.array([14,10])
                elif (y >= 0.3 and y < 0.4):
                    zone =  np.array([14,11])
                elif (y >= 0.4 and y < 0.5):
                    zone =  np.array([14,12])
                elif (y >= 0.5 and y < 0.6):
                    zone =  np.array([14,13])
                elif (y >= 0.6 and y < 0.7):
                    zone =  np.array([14,14])
                else:
                    zone =  np.array([14,15])
            elif (x >= 0.5 and x < 0.6):
                if (y >= 0 and y < 0.1):
                    zone =  np.array([15,8])
                elif (y >= 0.1 and y < 0.2):
                    zone =  np.array([15,9])
                elif (y >= 0.2 and y < 0.3):
                    zone =  np.array([15,10])
                elif (y >= 0.3 and y < 0.4):
                    zone =  np.array([15,11])
                elif (y >= 0.4 and y < 0.5):
                    zone =  np.array([15,12])
                elif (y >= 0.5 and y < 0.6):
                    zone =  np.array([15,13])
                elif (y >= 0.6 and y < 0.7):
                    zone =  np.array([15,14])
                else:
                    zone =  np.array([15,15])
            elif (x >= 0.6 and x < 0.7):
                if (y >= 0 and y < 0.1):
                    zone =  np.array([16,8])
                elif (y >= 0.1 and y < 0.2):
                    zone =  np.array([16,9])
                elif (y >= 0.2 and y < 0.3):
                    zone =  np.array([16,10])
                elif (y >= 0.3 and y < 0.4):
                    zone =  np.array([16,11])
                elif (y >= 0.4 and y < 0.5):
                    zone =  np.array([16,12])
                elif (y >= 0.5 and y < 0.6):
                    zone =  np.array([16,13])
                elif (y >= 0.6 and y < 0.7):
                    zone =  np.array([16,14])
                else:
                    zone =  np.array([16,15])
            elif (x >= 0.7 and x < 0.8):
                if (y >= 0 and y < 0.1):
                    zone =  np.array([17,8])
                elif (y >= 0.1 and y < 0.2):
                    zone =  np.array([17,9])
                elif (y >= 0.2 and y < 0.3):
                    zone =  np.array([17,10])
                elif (y >= 0.3 and y < 0.4):
                    zone =  np.array([17,11])
                elif (y >= 0.4 and y < 0.5):
                    zone =  np.array([17,12])
                elif (y >= 0.5 and y < 0.6):
                    zone =  np.array([17,13])
                elif (y >= 0.6 and y < 0.7):
                    zone =  np.array([17,14])
                else:
                    zone =  np.array([17,15])
            elif (x >= 0.8 and x < 0.9):
                if (y >= 0 and y < 0.1):
                    zone =  np.array([18,8])
                elif (y >= 0.1 and y < 0.2):
                    zone =  np.array([18,9])
                elif (y >= 0.2 and y < 0.3):
                    zone =  np.array([18,10])
                elif (y >= 0.3 and y < 0.4):
                    zone =  np.array([18,11])
                elif (y >= 0.4 and y < 0.5):
                    zone =  np.array([18,12])
                elif (y >= 0.5 and y < 0.6):
                    zone =  np.array([18,13])
                elif (y >= 0.6 and y < 0.7):
                    zone =  np.array([18,14])
                else:
                    zone =  np.array([18,15])
            else:
                if (y >= 0 and y < 0.1):
                    zone =  np.array([19,8])
                elif (y >= 0.1 and y < 0.2):
                    zone =  np.array([19,9])
                elif (y >= 0.2 and y < 0.3):
                    zone =  np.array([19,10])
                elif (y >= 0.3 and y < 0.4):
                    zone =  np.array([19,11])
                elif (y >= 0.4 and y < 0.5):
                    zone =  np.array([19,12])
                elif (y >= 0.5 and y < 0.6):
                    zone =  np.array([19,13])
                elif (y >= 0.6 and y < 0.7):
                    zone =  np.array([19,14])
                else:
                    zone =  np.array([19,15])
        else:
            if (x >= 0 and x < 0.1):
                if (y <= 0 and y > -0.1):
                    zone =  np.array([10,7])
                elif (y <= -0.1 and y > -0.2):
                    zone =  np.array([10,6])
                elif (y <= -0.2 and y > -0.3):
                    zone =  np.array([10,5])
                elif (y <= -0.3 and y > -0.4):
                    zone =  np.array([10,4])
                elif (y <= -0.4 and y > -0.5):
                    zone =  np.array([10,3])
                elif (y <= -0.5 and y > -0.6):
                    zone =  np.array([10,2])
                elif (y <= -0.6 and y > -0.7):
                    zone =  np.array([10,1])
                else:
                    zone =  np.array([10,0])
            elif (x >= 0.1 and x < 0.2):
                if (y <= 0 and y > -0.1):
                    zone =  np.array([11,7])
                elif (y <= -0.1 and y > -0.2):
                    zone =  np.array([11,6])
                elif (y <= -0.2 and y > -0.3):
                    zone =  np.array([11,5])
                elif (y <= -0.3 and y > -0.4):
                    zone =  np.array([11,4])
                elif (y <= -0.4 and y > -0.5):
                    zone =  np.array([11,3])
                elif (y <= -0.5 and y > -0.6):
                    zone =  np.array([11,2])
                elif (y <= -0.6 and y > -0.7):
                    zone =  np.array([11,1])
                else:
                    zone =  np.array([11,0])
            elif (x >= 0.2 and x < 0.3):
                if (y <= 0 and y > -0.1):
                    zone =  np.array([12,7])
                elif (y <= -0.1 and y > -0.2):
                    zone =  np.array([12,6])
                elif (y <= -0.2 and y > -0.3):
                    zone =  np.array([12,5])
                elif (y <= -0.3 and y > -0.4):
                    zone =  np.array([12,4])
                elif (y <= -0.4 and y > -0.5):
                    zone =  np.array([12,3])
                elif (y <= -0.5 and y > -0.6):
                    zone =  np.array([12,2])
                elif (y <= -0.6 and y > -0.7):
                    zone =  np.array([12,1])
                else:
                    zone =  np.array([12,0])
            elif (x >= 0.3 and x < 0.4):
                if (y <= 0 and y > -0.1):
                    zone =  np.array([13,7])
                elif (y <= -0.1 and y > -0.2):
                    zone =  np.array([13,6])
                elif (y <= -0.2 and y > -0.3):
                    zone =  np.array([13,5])
                elif (y <= -0.3 and y > -0.4):
                    zone =  np.array([13,4])
                elif (y <= -0.4 and y > -0.5):
                    zone =  np.array([13,3])
                elif (y <= -0.5 and y > -0.6):
                    zone =  np.array([13,2])
                elif (y <= -0.6 and y > -0.7):
                    zone =  np.array([13,1])
                else:
                    zone =  np.array([13,0])
            elif (x >= 0.4 and x < 0.5):
                if (y <= 0 and y > -0.1):
                    zone =  np.array([14,7])
                elif (y <= -0.1 and y > -0.2):
                    zone =  np.array([14,6])
                elif (y <= -0.2 and y > -0.3):
                    zone =  np.array([14,5])
                elif (y <= -0.3 and y > -0.4):
                    zone =  np.array([14,4])
                elif (y <= -0.4 and y > -0.5):
                    zone =  np.array([14,3])
                elif (y <= -0.5 and y > -0.6):
                    zone =  np.array([14,2])
                elif (y <= -0.6 and y > -0.7):
                    zone =  np.array([14,1])
                else:
                    zone =  np.array([14,0])
            elif (x >= 0.5 and x < 0.6):
                if (y <= 0 and y > -0.1):
                    zone =  np.array([15,7])
                elif (y <= -0.1 and y > -0.2):
                    zone =  np.array([15,6])
                elif (y <= -0.2 and y > -0.3):
                    zone =  np.array([15,5])
                elif (y <= -0.3 and y > -0.4):
                    zone =  np.array([15,4])
                elif (y <= -0.4 and y > -0.5):
                    zone =  np.array([15,3])
                elif (y <= -0.5 and y > -0.6):
                    zone =  np.array([15,2])
                elif (y <= -0.6 and y > -0.7):
                    zone =  np.array([15,1])
                else:
                    zone =  np.array([15,0])
            elif (x >= 0.6 and x < 0.7):
                if (y <= 0 and y > -0.1):
                    zone =  np.array([16,7])
                elif (y <= -0.1 and y > -0.2):
                    zone =  np.array([16,6])
                elif (y <= -0.2 and y > -0.3):
                    zone =  np.array([16,5])
                elif (y <= -0.3 and y > -0.4):
                    zone =  np.array([16,4])
                elif (y <= -0.4 and y > -0.5):
                    zone =  np.array([16,3])
                elif (y <= -0.5 and y > -0.6):
                    zone =  np.array([16,2])
                elif (y <= -0.6 and y > -0.7):
                    zone =  np.array([16,1])
                else:
                    zone =  np.array([16,0])
            elif (x >= 0.7 and x < 0.8):
                if (y <= 0 and y > -0.1):
                    zone =  np.array([17,7])
                elif (y <= -0.1 and y > -0.2):
                    zone =  np.array([17,6])
                elif (y <= -0.2 and y > -0.3):
                    zone =  np.array([17,5])
                elif (y <= -0.3 and y > -0.4):
                    zone =  np.array([17,4])
                elif (y <= -0.4 and y > -0.5):
                    zone =  np.array([17,3])
                elif (y <= -0.5 and y > -0.6):
                    zone =  np.array([17,2])
                elif (y <= -0.6 and y > -0.7):
                    zone =  np.array([17,1])
                else:
                    zone =  np.array([17,0])
            elif (x >= 0.8 and x < 0.9):
                if (y <= 0 and y > -0.1):
                    zone =  np.array([18,7])
                elif (y <= -0.1 and y > -0.2):
                    zone =  np.array([18,6])
                elif (y <= -0.2 and y > -0.3):
                    zone =  np.array([18,5])
                elif (y <= -0.3 and y > -0.4):
                    zone =  np.array([18,4])
                elif (y <= -0.4 and y > -0.5):
                    zone =  np.array([18,3])
                elif (y <= -0.5 and y > -0.6):
                    zone =  np.array([18,2])
                elif (y <= -0.6 and y > -0.7):
                    zone =  np.array([18,1])
                else:
                    zone =  np.array([18,0])
            else:
                if (y <= 0 and y > -0.1):
                    zone =  np.array([19,7])
                elif (y <= -0.1 and y > -0.2):
                    zone =  np.array([19,6])
                elif (y <= -0.2 and y > -0.3):
                    zone =  np.array([19,5])
                elif (y <= -0.3 and y > -0.4):
                    zone =  np.array([19,4])
                elif (y <= -0.4 and y > -0.5):
                    zone =  np.array([19,3])
                elif (y <= -0.5 and y > -0.6):
                    zone =  np.array([19,2])
                elif (y <= -0.6 and y > -0.7):
                    zone =  np.array([19,1])
                else:
                    zone =  np.array([19,0])
    else:
        if (y >= 0 and y <= 0.8):
            if (x <= 0 and x > -0.1):
                if (y >= 0 and y < 0.1):
                    zone = np.array([9,8])
                elif (y >= 0.1 and y < 0.2):
                    zone = np.array([9,9])
                elif (y >= 0.2 and y < 0.3):
                    zone = np.array([9,10])
                elif (y >= 0.3 and y < 0.4):
                    zone = np.array([9,11])
                elif (y >= 0.4 and y < 0.5):
                    zone = np.array([9,12])
                elif (y >= 0.5 and y < 0.6):
                    zone = np.array([9,13])
                elif (y >= 0.6 and y < 0.7):
                    zone = np.array([9,14])
                else:
                    zone =  np.array([9,15])
            elif (x <= -0.1 and x > -0.2):
                if (y >= 0 and y < 0.1):
                    zone = np.array([8,8])
                elif (y >= 0.1 and y < 0.2):
                    zone = np.array([8,9])
                elif (y >= 0.2 and y < 0.3):
                    zone = np.array([8,10])
                elif (y >= 0.3 and y < 0.4):
                    zone = np.array([8,11])
                elif (y >= 0.4 and y < 0.5):
                    zone = np.array([8,12])
                elif (y >= 0.5 and y < 0.6):
                    zone = np.array([8,13])
                elif (y >= 0.6 and y < 0.7):
                    zone = np.array([8,14])
                else:
                    zone =  np.array([8,15])
            elif (x <= -0.2 and x > -0.3):
                if (y >= 0 and y < 0.1):
                    zone = np.array([7,8])
                elif (y >= 0.1 and y < 0.2):
                    zone = np.array([7,9])
                elif (y >= 0.2 and y < 0.3):
                    zone = np.array([7,10])
                elif (y >= 0.3 and y < 0.4):
                    zone = np.array([7,11])
                elif (y >= 0.4 and y < 0.5):
                    zone = np.array([7,12])
                elif (y >= 0.5 and y < 0.6):
                    zone = np.array([7,13])
                elif (y >= 0.6 and y < 0.7):
                    zone = np.array([7,14])
                else:
                    zone =  np.array([7,15])
            elif (x <= -0.3 and x > -0.4):
                if (y >= 0 and y < 0.1):
                    zone = np.array([6,8])
                elif (y >= 0.1 and y < 0.2):
                    zone = np.array([6,9])
                elif (y >= 0.2 and y < 0.3):
                    zone = np.array([6,10])
                elif (y >= 0.3 and y < 0.4):
                    zone = np.array([6,11])
                elif (y >= 0.4 and y < 0.5):
                    zone = np.array([6,12])
                elif (y >= 0.5 and y < 0.6):
                    zone = np.array([6,13])
                elif (y >= 0.6 and y < 0.7):
                    zone = np.array([6,14])
                else:
                    zone =  np.array([6,15])
            elif (x <= -0.4 and x > -0.5):
                if (y >= 0 and y < 0.1):
                    zone = np.array([5,8])
                elif (y >= 0.1 and y < 0.2):
                    zone = np.array([5,9])
                elif (y >= 0.2 and y < 0.3):
                    zone = np.array([5,10])
                elif (y >= 0.3 and y < 0.4):
                    zone = np.array([5,11])
                elif (y >= 0.4 and y < 0.5):
                    zone = np.array([5,12])
                elif (y >= 0.5 and y < 0.6):
                    zone = np.array([5,13])
                elif (y >= 0.6 and y < 0.7):
                    zone = np.array([5,14])
                else:
                    zone =  np.array([5,15])
            elif (x <= -0.5 and x > -0.6):
                if (y >= 0 and y < 0.1):
                    zone = np.array([4,8])
                elif (y >= 0.1 and y < 0.2):
                    zone = np.array([4,9])
                elif (y >= 0.2 and y < 0.3):
                    zone = np.array([4,10])
                elif (y >= 0.3 and y < 0.4):
                    zone = np.array([4,11])
                elif (y >= 0.4 and y < 0.5):
                    zone = np.array([4,12])
                elif (y >= 0.5 and y < 0.6):
                    zone = np.array([4,13])
                elif (y >= 0.6 and y < 0.7):
                    zone = np.array([4,14])
                else:
                    zone =  np.array([4,15])
            elif (x <= -0.6 and x > -0.7):
                if (y >= 0 and y < 0.1):
                    zone = np.array([3,8])
                elif (y >= 0.1 and y < 0.2):
                    zone = np.array([3,9])
                elif (y >= 0.2 and y < 0.3):
                    zone = np.array([3,10])
                elif (y >= 0.3 and y < 0.4):
                    zone = np.array([3,11])
                elif (y >= 0.4 and y < 0.5):
                    zone = np.array([3,12])
                elif (y >= 0.5 and y < 0.6):
                    zone = np.array([3,13])
                elif (y >= 0.6 and y < 0.7):
                    zone = np.array([3,14])
                else:
                    zone =  np.array([3,15])
            elif (x <= -0.7 and x > -0.8):
                if (y >= 0 and y < 0.1):
                    zone = np.array([2,8])
                elif (y >= 0.1 and y < 0.2):
                    zone = np.array([2,9])
                elif (y >= 0.2 and y < 0.3):
                    zone = np.array([2,10])
                elif (y >= 0.3 and y < 0.4):
                    zone = np.array([2,11])
                elif (y >= 0.4 and y < 0.5):
                    zone = np.array([2,12])
                elif (y >= 0.5 and y < 0.6):
                    zone = np.array([2,13])
                elif (y >= 0.6 and y < 0.7):
                    zone = np.array([2,14])
                else:
                    zone =  np.array([2,15])
            elif (x <= -0.8 and x > -0.9):
                if (y >= 0 and y < 0.1):
                    zone = np.array([1,8])
                elif (y >= 0.1 and y < 0.2):
                    zone = np.array([1,9])
                elif (y >= 0.2 and y < 0.3):
                    zone = np.array([1,10])
                elif (y >= 0.3 and y < 0.4):
                    zone = np.array([1,11])
                elif (y >= 0.4 and y < 0.5):
                    zone = np.array([1,12])
                elif (y >= 0.5 and y < 0.6):
                    zone = np.array([1,13])
                elif (y >= 0.6 and y < 0.7):
                    zone = np.array([1,14])
                else:
                    zone =  np.array([1,15])
            else:
                if (y >= 0 and y < 0.1):
                    zone = np.array([0,8])
                elif (y >= 0.1 and y < 0.2):
                    zone = np.array([0,9])
                elif (y >= 0.2 and y < 0.3):
                    zone = np.array([0,10])
                elif (y >= 0.3 and y < 0.4):
                    zone = np.array([0,11])
                elif (y >= 0.4 and y < 0.5):
                    zone = np.array([0,12])
                elif (y >= 0.5 and y < 0.6):
                    zone = np.array([0,13])
                elif (y >= 0.6 and y < 0.7):
                    zone = np.array([0,14])
                else:
                    zone =  np.array([0,15])
        else:
            if (x <= 0 and x > -0.1):
                if (y <= 0 and y > -0.1):
                    zone = np.array([9,7])
                elif (y <= -0.1 and y > -0.2):
                    zone = np.array([9,6])
                elif (y <= -0.2 and y > -0.3):
                    zone = np.array([9,5])
                elif (y <= -0.3 and y > -0.4):
                    zone = np.array([9,4])
                elif (y <= -0.4 and y > -0.5):
                    zone = np.array([9,3])
                elif (y <= -0.5 and y > -0.6):
                    zone = np.array([9,2])
                elif (y <= -0.6 and y > -0.7):
                    zone = np.array([9,1])
                else:
                    zone = np.array([9,0])
            elif (x <= -0.1 and x > -0.2):
                if (y <= 0 and y > -0.1):
                    zone = np.array([8,7])
                elif (y <= -0.1 and y > -0.2):
                    zone = np.array([8,6])
                elif (y <= -0.2 and y > -0.3):
                    zone = np.array([8,5])
                elif (y <= -0.3 and y > -0.4):
                    zone = np.array([8,4])
                elif (y <= -0.4 and y > -0.5):
                    zone = np.array([8,3])
                elif (y <= -0.5 and y > -0.6):
                    zone = np.array([8,2])
                elif (y <= -0.6 and y > -0.7):
                    zone = np.array([8,1])
                else:
                    zone = np.array([8,0])
            elif (x <= -0.2 and x > -0.3):
                if (y <= 0 and y > -0.1):
                    zone = np.array([7,7])
                elif (y <= -0.1 and y > -0.2):
                    zone = np.array([7,6])
                elif (y <= -0.2 and y > -0.3):
                    zone = np.array([7,5])
                elif (y <= -0.3 and y > -0.4):
                    zone = np.array([7,4])
                elif (y <= -0.4 and y > -0.5):
                    zone = np.array([7,3])
                elif (y <= -0.5 and y > -0.6):
                    zone = np.array([7,2])
                elif (y <= -0.6 and y > -0.7):
                    zone = np.array([7,1])
                else:
                    zone = np.array([7,0])
            elif (x <= -0.3 and x > -0.4):
                if (y <= 0 and y > -0.1):
                    zone = np.array([6,7])
                elif (y <= -0.1 and y > -0.2):
                    zone = np.array([6,6])
                elif (y <= -0.2 and y > -0.3):
                    zone = np.array([6,5])
                elif (y <= -0.3 and y > -0.4):
                    zone = np.array([6,4])
                elif (y <= -0.4 and y > -0.5):
                    zone = np.array([6,3])
                elif (y <= -0.5 and y > -0.6):
                    zone = np.array([6,2])
                elif (y <= -0.6 and y > -0.7):
                    zone = np.array([6,1])
                else:
                    zone = np.array([6,0])
            elif (x <= -0.4 and x > -0.5):
                if (y <= 0 and y > -0.1):
                    zone = np.array([5,7])
                elif (y <= -0.1 and y > -0.2):
                    zone = np.array([5,6])
                elif (y <= -0.2 and y > -0.3):
                    zone = np.array([5,5])
                elif (y <= -0.3 and y > -0.4):
                    zone = np.array([5,4])
                elif (y <= -0.4 and y > -0.5):
                    zone = np.array([5,3])
                elif (y <= -0.5 and y > -0.6):
                    zone = np.array([5,2])
                elif (y <= -0.6 and y > -0.7):
                    zone = np.array([5,1])
                else:
                    zone = np.array([5,0])
            elif (x <= -0.5 and x > -0.6):
                if (y <= 0 and y > -0.1):
                    zone = np.array([4,7])
                elif (y <= -0.1 and y > -0.2):
                    zone = np.array([4,6])
                elif (y <= -0.2 and y > -0.3):
                    zone = np.array([4,5])
                elif (y <= -0.3 and y > -0.4):
                    zone = np.array([4,4])
                elif (y <= -0.4 and y > -0.5):
                    zone = np.array([4,3])
                elif (y <= -0.5 and y > -0.6):
                    zone = np.array([4,2])
                elif (y <= -0.6 and y > -0.7):
                    zone = np.array([4,1])
                else:
                    zone = np.array([4,0])
            elif (x <= -0.6 and x > -0.7):
                if (y <= 0 and y > -0.1):
                    zone = np.array([3,7])
                elif (y <= -0.1 and y > -0.2):
                    zone = np.array([3,6])
                elif (y <= -0.2 and y > -0.3):
                    zone = np.array([3,5])
                elif (y <= -0.3 and y > -0.4):
                    zone = np.array([3,4])
                elif (y <= -0.4 and y > -0.5):
                    zone = np.array([3,3])
                elif (y <= -0.5 and y > -0.6):
                    zone = np.array([3,2])
                elif (y <= -0.6 and y > -0.7):
                    zone = np.array([3,1])
                else:
                    zone = np.array([3,0])
            elif (x <= -0.7 and x > -0.8):
                if (y <= 0 and y > -0.1):
                    zone = np.array([2,7])
                elif (y <= -0.1 and y > -0.2):
                    zone = np.array([2,6])
                elif (y <= -0.2 and y > -0.3):
                    zone = np.array([2,5])
                elif (y <= -0.3 and y > -0.4):
                    zone = np.array([2,4])
                elif (y <= -0.4 and y > -0.5):
                    zone = np.array([2,3])
                elif (y <= -0.5 and y > -0.6):
                    zone = np.array([2,2])
                elif (y <= -0.6 and y > -0.7):
                    zone = np.array([2,1])
                else:
                    zone = np.array([2,0])
            elif (x <= -0.8 and x > -0.9):
                if (y <= 0 and y > -0.1):
                    zone = np.array([1,7])
                elif (y <= -0.1 and y > -0.2):
                    zone = np.array([1,6])
                elif (y <= -0.2 and y > -0.3):
                    zone = np.array([1,5])
                elif (y <= -0.3 and y > -0.4):
                    zone = np.array([1,4])
                elif (y <= -0.4 and y > -0.5):
                    zone = np.array([1,3])
                elif (y <= -0.5 and y > -0.6):
                    zone = np.array([1,2])
                elif (y <= -0.6 and y > -0.7):
                    zone = np.array([1,1])
                else:
                    zone = np.array([1,0])
            else:
                if (y <= 0 and y > -0.1):
                    zone = np.array([0,7])
                elif (y <= -0.1 and y > -0.2):
                    zone = np.array([0,6])
                elif (y <= -0.2 and y > -0.3):
                    zone = np.array([0,5])
                elif (y <= -0.3 and y > -0.4):
                    zone = np.array([0,4])
                elif (y <= -0.4 and y > -0.5):
                    zone = np.array([0,3])
                elif (y <= -0.5 and y > -0.6):
                    zone = np.array([0,2])
                elif (y <= -0.6 and y > -0.7):
                    zone = np.array([0,1])
                else:
                    zone = np.array([0,0])
    return zone

def get_direction(rad):
    deg = math.degrees(rad)
    if ((deg > 0 and deg < 7.5) or (deg > 352.5 and deg < 360)):
        direction = 'east'
    elif (deg > 7.5 and deg < 22.5):
        direction = 'east2'
    elif (deg > 22.5 and deg < 37.5):
        direction = 'east1'
    elif (deg > 37.5 and deg < 52.5):
        direction = 'north_east'
    elif (deg > 52.5 and deg < 67.5):
        direction = 'north_east2'
    elif (deg > 67.5 and deg < 82.5):
        direction = 'north_east1'
    elif (deg > 82.5 and deg < 97.5):
        direction = 'north'
    elif (deg > 97.5 and deg < 112.5):
        direction = 'north2'
    elif (deg > 112.5 and deg < 127.5):
        direction = 'north1'
    elif (deg > 127.5 and deg < 142.5):
        direction = 'north_west'
    elif (deg > 142.5 and deg < 157.5):
        direction = 'north_west2'
    elif (deg > 157.5 and deg < 172.5):
        direction = 'north_west1'
    elif (deg > 172.5 and deg < 187.5):
        direction = 'west'
    elif (deg > 187.5 and deg < 202.5):
        direction = 'west2'
    elif (deg > 202.5 and deg < 217.5):
        direction = 'west1'
    elif (deg > 217.5 and deg < 232.5):
        direction = 'south_west'
    elif (deg > 232.5 and deg < 247.5):
        direction = 'south_west2'
    elif (deg > 247.5 and deg < 262.5):
        direction = 'south_west1'
    elif (deg > 262.5 and deg < 277.5):
        direction = 'south'
    elif (deg > 277.5 and deg < 292.5):
        direction = 'south2'
    elif (deg > 292.5 and deg < 307.5):
        direction = 'south1'
    elif (deg > 307.5 and deg < 322.5):
        direction = 'south_east'
    elif (deg > 322.5 and deg < 337.5):
        direction = 'south_east2'
    elif (deg > 337.5 and deg < 352.5):
        direction = 'south_east1'
    else:
        direction = None
    return direction

def get_minmax_values(x_min, x_max, y_min, y_max, grid):
    if x_min == None:
        x_min = grid[0]
    elif x_min > grid[0]:
        x_min = grid[0]
    if x_max == None:
        x_max = grid[0]
    elif x_max < grid[0]:
        x_max = grid[0]
    if y_min == None:
        y_min = grid[1]
    elif y_min > grid[1]:
        y_min = grid[1]
    if y_max == None:
        y_max = grid[1] 
    elif y_max < grid[1]:
        y_max = grid[1]
    return x_min, x_max, y_min, y_max

def get_tri_pts_arr(x_pos, y_pos, ang):
    size = 0.05
    shootRad = 0.8
    shootWin = np.pi/4
    pt1 = [x_pos, y_pos]+size*np.array([np.cos(ang), np.sin(ang)])
    pt2 = pt1 + shootRad*np.array([np.cos(ang+shootWin/2), np.sin(ang+shootWin/2)])
    pt3 = pt1 + shootRad*np.array([np.cos(ang-shootWin/2), np.sin(ang-shootWin/2)])
    A = np.array([[pt1[0], pt2[0], pt3[0]],
                    [pt1[1], pt2[1], pt3[1]],
                    [     1,      1,      1]])       
    return(A)

def attacker_in_range(agent, A):
    if (agent[0]) == 1.0:
        b = np.array([[agent[1]],[agent[2]],[1]])
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
