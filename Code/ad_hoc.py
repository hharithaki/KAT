from csv import writer
import math
import subprocess
import re
import utils
import numpy as np
from sklweka.classifiers import Classifier
from sklweka.dataset import Instance, missing_value
try:
  import hfo
except ImportError:
  print('Failed to import hfo. To install hfo, in the HFO directory'\
    ' run: \"pip install .\"')
  exit()
pre_asp_learner = 'ASP/learner_pre.sp'
asp_learner = 'ASP/learner.sp'
display_marker = 'display'
# offense1_model_file = 'models/hlca.model'
# offense2_model_file = 'models/gt.model'
offense3_model_file = 'models/axiom300.model'
defense1_model_file = 'models/base300.model'
defense2_model_file = 'models/goalie300.model'

class Ad_Hoc_Agent:
    class Agent:
        x_pos: float = 0
        y_pos: float = 0
        grid: np.array = []
        orientation: float = 0
        ball_x: float = 0
        ball_y: float = 0
        proximity_op: float = 0
        goal_opening_angle: float = 0
        able_to_kick: float = 0
        dist_to_goal: float = 0
        goal_center_angle: float = 0
        last_action_succ: float = 0
        stamina: float = 0
        
        def __init__(self, array: list):
            self.x_pos = array[0]
            self.y_pos = array[1]
            self.grid = utils.get_gridno(array[0], array[1])
            self.orientation = array[2]
            self.ball_x = array[3]
            self.ball_y = array[4]
            self.able_to_kick = array[5]
            self.dist_to_goal = array[6]
            self.goal_center_angle = array[7]
            self.goal_opening_angle = array[8]
            self.proximity_op = array[9]
            self.last_action_succ = array[-2]
            self.stamina = array[-1]
        
        def to_array(self):
            return [self.x_pos, self.y_pos, self.grid, self.orientation, 
                    self.ball_x, self.ball_y, self.able_to_kick, self.dist_to_goal,
                    self.goal_center_angle, self.goal_opening_angle,
                    self.proximity_op, self.last_action_succ,
                    self.stamina]
        
        def len(self):
            return len(self.to_array())

    class Teammate:
        goal_angle: float = 0
        proximity_op: float = 0
        pass_angle: float = 0
        x_pos: float = 0
        y_pos: float = 0
        grid: np.array = []
        uniform_num: int = 0
        
        def __init__(self, goal_angle, proximity_op, pass_angle, x_pos, y_pos, uniform_num):
            self.goal_angle = goal_angle
            self.proximity_op = proximity_op
            self.pass_angle = pass_angle
            self.x_pos = x_pos
            self.y_pos = y_pos
            self.grid = utils.get_gridno(x_pos, y_pos)
            self.uniform_num = uniform_num
        
        def to_array(self):
            return [self.goal_angle, self.proximity_op, self.pass_angle, self.x_pos, self.y_pos, self.grid, self.uniform_num]
        
        def len(self):
            return len(self.to_array())
    
    class Opponent:
        x_pos: float = 0
        y_pos: float = 0
        grid: np.array = []
        uniform_num: int = 0
        
        def __init__(self, array: list):
            self.x_pos = array[0]
            self.y_pos = array[1]
            self.grid = utils.get_gridno(array[0], array[1])
            self.uniform_num = array[2]
        
        def to_array(self):
            return [self.x_pos, self.y_pos, self.grid, self.uniform_num]
        
        def len(self):
            return len(self.to_array())

    def __init__(self, num_teammates, num_opponents):
        self.num_teammates = num_teammates
        self.num_opponents = num_opponents
        self.current_offense_model = [offense3_model_file]
        self.current_defense_model = [defense1_model_file, defense2_model_file]
        self.agent = None
        self.teammates = None
        self.opponents = None
        self.last_answer_set = None
        self.locations = None
        self.uniform_numbers = None
        self.data = []
        self.previous_actions = None
        self.score = 0
        self.count = 0
        self.previous_real_action = 0

    # Decide the action to be taken by the agent 
    def get_action(self, state, step):
        # self.data.append(state)
        # if len(self.data) > 5:
        #     self.data = self.data[1:]
        # if step > 2:
        #     real_action = utils.update_models(self.data, 'offense')
        #     print('-----------------------------------', step, real_action)
        #     if self.previous_actions is not None:
        #         with open('axiom-test-4v5.csv', 'a') as file_object:
        #             writer_object = writer(file_object)
        #             row = []
        #             for i in range(len(state)):
        #                 row.append(state[i])
        #             row.append(self.previous_real_action) 
        #             writer_object.writerow(row)
        #             self.previous_real_action = real_action
        if int(state[5]) == 1: 
            reader = open(pre_asp_learner, 'r')
            pre_asp = reader.read()
            reader.close()
            pre_asp_split = pre_asp.split('\n')
            display_marker_index = pre_asp_split.index(display_marker)
            input, input1 = self.get_actions_terms(state)
            answer_list = self.run_ASP_learner(input, input1, pre_asp_split, display_marker_index)
            if len(answer_list) <= 1:
                answer_list = self.run_ASP_learner(input, input1=[], pre_asp_split=pre_asp_split, display_marker_index=display_marker_index)
            actions = self.process_answerlist(answer_list)
        else:
            agent_pos = np.array([state[0], state[1]])
            ball_pos = np.array([state[3], state[4]])
            teammates_pos = [np.array([state[19], state[20]]), np.array([state[22], state[23]]), np.array([state[25], state[26]])]
            team_ball_dist = []
            teammate_y = []
            for teammate in teammates_pos:
                team_ball_dist.append(np.linalg.norm(teammate - ball_pos))
                teammate_y.append(teammate[1])
            if any(num < 0.1 for num in team_ball_dist):
                # teamamte has the ball, come up with a better positioning
                next_actions = self.get_action_terms(state)
                if any(y < 0 for y in teammate_y):
                    actions = (5, 0.4, 0.3)
                else:
                    actions = (5, 0.4, -0.3)
            else:
                actions = 8 # 8 for aut and axiom?others 7
        return actions

    # Look ahead using the models
    def get_actions_terms(self, state):
        input = self.get_terms(state, 0, input = []) # first step
        input1 = []
        for timestep in range(1):
            future_actions = self.predict_class(state)
            self.previous_actions = future_actions
            if int(state[5]) == 1:
                future_actions[0] = str(hfo.DRIBBLE) # if you change this -> chnge this inside run_environemnt as well
            else:
                future_actions[0] = str(hfo.MOVE)
            self.locations, self.uniform_numbers = utils.run_environment(state, future_actions)
            input1 = self.get_future_terms(timestep+1, input1)
        return input, input1

    # Look ahead using the models
    def get_action_terms(self, state):
        for timestep in range(1):
            future_actions = self.predict_class(state)
            if int(state[5]) == 1:
                future_actions[0] = str(hfo.DRIBBLE) # if you change this -> chnge this inside run_environemnt as well
            else:
                future_actions[0] = str(hfo.MOVE)
        return future_actions

    # Sort and map answer set to actions
    def process_answerlist(self, answer_list):
        actions = []
        if len(answer_list) > 1: # removed situations = *satisfied *no of steps is not enough(empty (NOTE:not{})) *too many answers(error)
            action_list = []
            answer = utils.process_answer(answer_list)
            answer_split = answer.strip('{}\n').split(', ')
            for element in answer_split:
                for i in range(len(answer_split)):
                    if re.search(rf',{i}\)$',element) != None:
                        action_list.insert(i, element)
            actions = utils.map_actions(action_list, self.agent, self.teammates, self.opponents) if len(action_list) > 0 else [0]
        else: # for now seems like each action take 2 runs
            actions = utils.get_action(self.agent, self.teammates, self.opponents, self.locations, self.uniform_numbers)
        return actions[0]

    # return answer sets for the new ASP file
    def run_ASP_learner(self, input, input1, pre_asp_split, display_marker_index):
        opponents_inside_goal, x_min, x_max, y_min, y_max, prefix = utils.get_limits(self.agent, self.teammates[0], self.opponents, self.locations, self.uniform_numbers)
        if self.agent.grid[0] < 13:
            goal_term = ['goal(I) :- holds(ball_in(13,Y),I).'] # reach region that can shoot - it wont pass here (need to add some axioms to make it pass to teammate if offense close)
            angle_terms = []
        else:
            goal_term = ['goal(I) :- holds(score_goal,I).']
            angle_terms = utils.get_angle_terms(self.agent, opponents_inside_goal)
        next_terms = utils.get_next_terms(x_min, x_max, y_min, y_max)
        asp_split = prefix + pre_asp_split[:display_marker_index] + goal_term + next_terms + angle_terms + input + input1 + pre_asp_split[display_marker_index:]
        asp = '\n'.join(asp_split)
        f1 = open(asp_learner, 'w')
        f1.write(asp)
        f1.close()
        answer = subprocess.check_output('java -jar ASP/sparc.jar ' +asp_learner+' -A',shell=True)
        answer_split = (answer.decode('ascii'))
        return answer_split

    # process obs and set the fluents that need to be included and considered in the new ASP program
    def get_terms(self, state, step, input):
        self.agent = self.Agent(state)
        ball_grid = utils.get_gridno(self.agent.ball_x, self.agent.ball_y)
        input = utils.get_ASP_terms(input, 'learner', step, self.agent.grid, ball_grid, self.agent.able_to_kick)

        self.teammates = self.get_teammates(state[10:], self.num_teammates)
        idx = 10 + self.num_teammates * 6
        self.opponents = self.get_opponents(state[idx:], self.num_opponents)
        input = utils.get_ASP_terms_other(input, step, self.agent, self.teammates, self.opponents)
        return input

    def get_future_terms(self, step, input):
        input = utils.get_future_ASP_terms(input, step, self.agent, self.locations, self.uniform_numbers, self.teammates[0].uniform_num)
        return input

    # return the predicted actions for the other agents as a list/array
    def predict_class(self, state):
        predict_actions = ['act0']
        for agent in range(self.num_teammates + self.num_opponents):
            values = utils.process_data(state, agent)
            if agent < self.num_teammates:
                model, header = Classifier.deserialize(offense3_model_file)
            elif agent == self.num_teammates:
                model, header = Classifier.deserialize(defense2_model_file) # goalie
            else:
                model, header = Classifier.deserialize(defense1_model_file)
            # create new instance
            inst = Instance.create_instance(values)
            inst.dataset = header
            # make prediction
            index = model.classify_instance(inst)
            predict_actions.append(header.class_attribute.value(int(index)))
        return predict_actions
    
    def get_teammates(self, array: list, num_teammates: int) -> list:
        teammates_goal_opening_angle = array[:num_teammates]
        proximities_to_opponents = array[num_teammates:num_teammates * 2]
        pass_opening_angles = array[num_teammates * 2:num_teammates * 3]

        array = array[num_teammates*3:]
        teammates_dict = dict()
        for t_idx in range(num_teammates):
            aux_array = array[t_idx * 3:]
            x_pos = aux_array[0]
            y_pos = aux_array[1]
            uniform_number = aux_array[2]
            tt = self.Teammate(
                goal_angle=teammates_goal_opening_angle[t_idx],
                proximity_op=proximities_to_opponents[t_idx],
                pass_angle=pass_opening_angles[t_idx],
                x_pos=x_pos, y_pos=y_pos, uniform_num=uniform_number
            )
            teammates_dict[uniform_number] = tt

        teammates_list = []
        for t_idx, uniform_num in enumerate(sorted(teammates_dict.keys())):
            teammates_list.append(teammates_dict[uniform_num])
        return teammates_list
    
    def get_opponents(self, array: np.ndarray, num_op: int) -> list:
        opponents_dict = dict()
        for op in range(0, num_op):
            index = 3 * op
            uni = array[index + 2]
            opponents_dict[uni] = self.Opponent(array[index: index + 3])
        return [opponents_dict[uni] for uni in sorted(opponents_dict.keys())]
