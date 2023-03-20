#const n = 8.
sorts
#x_value = 11..19.
#y_value = 5..11.
#dist = 0..19.
#agent = {learner}.
#teammate = {offense2}. % , offense3
#offense = #agent + #teammate.
#defense = {defense1 , defense2, defense3, defense4, defense5}.
#other_agents = #offense + #defense.
#all_agents = #agent + #other_agents.
#angle_val = 0..180.% {0, 1, }.
#step = 0..n.
#boolean = {true, false}.

#agent_actions = shoot(#agent) + pass(#agent, #teammate) + dribble(#agent, #x_value, #y_value) + move(#agent, #x_value, #y_value).

#action = #agent_actions.

% ball_in(#dist, #dist) should be changed to ball_in(#x_value, #y_value) one conditions added
#inertial_f = {score_goal} + ball_in(#dist, #dist) + has_ball(#all_agents) + in(#agent, #x_value, #y_value).
#defined_f = {far_from_goal} + should_pass(#teammate) + should_pass_position(#teammate) + def_close(#teammate) + def_too_close(#agent) 
            + agent_in(#other_agents, #x_value, #y_value) + agent_angle(#angle_val)
            + distance_lea_to_def(#defense, #dist, #dist).
#fluents = #inertial_f + #defined_f.

predicates
angle(#angle_val, #x_value, #y_value, #x_value, #y_value).
next_to(#x_value, #y_value, #x_value, #y_value).
holds(#fluents, #step).
occurs(#action, #step).

%%history
hpd(#agent_actions, #step).
obs(#fluents, #boolean, #step).
current_step(#step).
% unobserved(#exogenous_actions, #step).
diagnosing(#step).

%%planning
success().
something_happened(#step).
goal(#step).
planning(#step).

rules
% -------------------------- casual laws --------------------------%

% move causes the agent to be in that place
holds(in(R,X,Y),I+1) :- occurs(move(R,X,Y),I).

% dribble cause the agent and ball to be in that place
holds(in(R,X,Y),I+1) :- occurs(dribble(R,X,Y),I).
holds(ball_in(X,Y),I+1) :- occurs(dribble(R,X,Y),I).

% pass cause the new agent to have the ball and the passing agent to lose the ball
-holds(has_ball(R),I+1) :- occurs(pass(R,O),I).
holds(has_ball(O),I+1) :- occurs(pass(R,O),I).
holds(score_goal,I+1) :- occurs(pass(R,O),I), holds(should_pass(O),I). % enourage pass
holds(ball_in(13,6),I+1) :- occurs(pass(R,O),I), holds(should_pass_position(O),I). % 6 is a dummy value

% shoot cause agent to score a goal
holds(score_goal,I+1) :- occurs(shoot(R),I), holds(agent_angle(V),I), V >= 19.

% ----------------------- state constraints -----------------------%

% next_to works both ways. If not specified then not next_to each other
next_to(X1,Y1,X2,Y2) :- next_to(X2,Y2,X1,Y1).
-next_to(X1,Y1,X2,Y2) :- not next_to(X1,Y1,X2,Y2). 

% an agent cannot be in two places at the same time.
-holds(in(R,X1,Y1),I) :- holds(in(R,X2,Y2),I), X1 != X2, Y1 != Y2.
-holds(in(R,X1,Y1),I) :- holds(in(R,X2,Y2),I), X1 != X2, Y1 = Y2.
-holds(in(R,X1,Y1),I) :- holds(in(R,X2,Y2),I), X1 = X2, Y1 != Y2.

% two agents cannot have the ball at the same time
-holds(has_ball(A1),I) :- holds(has_ball(A2),I), A1 != A2.

% ball cannot be in two places at the same time
-holds(ball_in(X1,Y1),I) :- holds(ball_in(X2,Y2),I), X1 != X2, Y1 != Y2.
-holds(ball_in(X1,Y1),I) :- holds(ball_in(X2,Y2),I), X1 != X2, Y1 = Y2.
-holds(ball_in(X1,Y1),I) :- holds(ball_in(X2,Y2),I), X1 = X2, Y1 != Y2.

% if other agent has the ball then the ball is in the location of that agent
% holds(ball_in(X,Y),I) :- holds(has_ball(A),I), holds(agent_in(A,X,Y),I). messes should_pass_position

%% updating defined fleuents %%
holds(agent_in(A,X,Y),I+1) :- holds(agent_in(A,X,Y),I), not holds(agent_in(A,X1,Y1),I+1), I >= 1, X1!=X,Y1!=Y.

holds(distance_lea_to_def(A,XA-X,YA-Y),I) :- holds(in(learner,X,Y),I), holds(agent_in(A,XA,YA),I), #defense(A), XA >= X, YA >= Y.
holds(distance_lea_to_def(A,X-XA,Y-YA),I) :- holds(in(learner,X,Y),I), holds(agent_in(A,XA,YA),I), #defense(A), XA <= X, YA <= Y.
holds(distance_lea_to_def(A,XA-X,Y-YA),I) :- holds(in(learner,X,Y),I), holds(agent_in(A,XA,YA),I), #defense(A), XA >= X, YA <= Y.
holds(distance_lea_to_def(A,X-XA,YA-Y),I) :- holds(in(learner,X,Y),I), holds(agent_in(A,XA,YA),I), #defense(A), XA <= X, YA >= Y.

% if the distane from the agent to nearest opponent is less than 1 grid cell and 
% if the distanc from the teammte to the nearest opponent is more than 3 grid cells consider passing to the nearest teammate
% for now lets make this 'pass' a must => depending on the results we will change this later
holds(def_too_close(O),I) :- #defense(D), #agent(O), holds(distance_lea_to_def(D,X,Y),I), X <= 1, Y <= 1.

holds(should_pass(T),I) :- holds(def_too_close(learner),I), not holds(agent_angle(V),I), V>=19, #teammate(T).
holds(should_pass_position(T),I) :- holds(def_too_close(learner),I), #teammate(T).

% update far from goal or not
holds(far_from_goal,I) :- holds(in(R,X,Y),I), X < 13.
holds(agent_angle(V),I) :- angle(V,XO,YO,X,Y), holds(agent_in(D,XO,YO),I), #defense(D), holds(in(learner,X,Y),I).

% ------------------------ inertial axioms ------------------------%

holds(F,I+1) :- #inertial_f(F), holds(F,I), not -holds(F,I+1).
-holds(F,I+1) :- #inertial_f(F), -holds(F,I), not holds(F,I+1).

% ------------------------------ CWA ------------------------------%

-occurs(A,I) :- not occurs(A,I).
-holds(F,I) :- #defined_f(F), not holds(F,I).

% -------------------- executability conditions -------------------%

% impossible to move to a location it is already in
-occurs(move(R,X,Y),I) :- holds(in(R,X,Y),I). 

% impossible to dribble to a location it is already in
-occurs(dribble(R,X,Y),I) :- holds(in(R,X,Y),I).

% impossible to move between locations which are not next to each other
-occurs(move(R,X1,Y1),I) :- holds(in(R,X2,Y2),I), -next_to(X1,Y1,X2,Y2).

% impossible to dribble between locations which are not next to each other
-occurs(dribble(R,X1,Y1),I) :- holds(in(R,X2,Y2),I), -next_to(X1,Y1,X2,Y2).

% impossible to move if the agent has the ball - have to dribble
-occurs(move(R,X,Y),I) :- holds(has_ball(R),I).

% impossible to dribble if the agent does not has the ball - have to move
-occurs(dribble(R,X,Y),I) :- not holds(has_ball(R),I).

% impossible to pass if the agent does not have the ball
-occurs(pass(R,O),I) :- not holds(has_ball(R),I).

% impossible to shoot if the agent does not have the ball
-occurs(shoot(R),I) :- not holds(has_ball(R),I).

% impossible to shoot if the agent is far from goal i
-occurs(shoot(R),I) :- holds(far_from_goal,I).

% impossible to shoot if not safe
% -occurs(shoot(R),I) :- holds(should_pass(T),I).
% -occurs(dribble(R,X,Y),I) :- holds(should_pass_position(T),I).

% --------------------------- planning ---------------------------%

% to achieve success the system should satisfies the goal. Failure is not acceptable
success :- goal(I), I <= n.
:- not success, current_step(I0), planning(I0).

% consider the occurrence of exogenous actions when they are absolutely necessary for resolving a conflict
occurs(A,I) :+ #agent_actions(A), #step(I), current_step(I0), planning(I0), I0 <= I.

% agent can not execute parallel actions
-occurs(A1,I) :- occurs(A2,I), A1 != A2, #agent_actions(A1), #agent_actions(A2).

% an action should occur at each time step until the goal is achieved
something_happened(I1) :- current_step(I0), planning(I0), I0 <= I1, occurs(A,I1), #agent_actions(A).
:- not something_happened(I), something_happened(I+1), I0 <= I, current_step(I0), planning(I0).

%%%--------------------------------------------------------------%%%

planning(I) :- current_step(I).

planning(0).
current_step(0).

% --------------- %

goal(I) :- holds(score_goal,I).
next_to(12,5,12,6).
next_to(12,5,13,5).
next_to(13,5,13,6).
next_to(13,5,14,5).
next_to(14,5,14,6).
next_to(14,5,15,5).
next_to(15,5,15,6).
next_to(15,5,16,5).
next_to(16,5,16,6).
next_to(16,5,17,5).
next_to(17,5,17,6).
next_to(17,5,18,5).
next_to(18,5,18,6).
next_to(18,5,19,5).
next_to(19,5,19,6).
next_to(12,6,12,7).
next_to(12,6,13,6).
next_to(13,6,13,7).
next_to(13,6,14,6).
next_to(14,6,14,7).
next_to(14,6,15,6).
next_to(15,6,15,7).
next_to(15,6,16,6).
next_to(16,6,16,7).
next_to(16,6,17,6).
next_to(17,6,17,7).
next_to(17,6,18,6).
next_to(18,6,18,7).
next_to(18,6,19,6).
next_to(19,6,19,7).
next_to(12,7,13,7).
next_to(13,7,14,7).
next_to(14,7,15,7).
next_to(15,7,16,7).
next_to(16,7,17,7).
next_to(17,7,18,7).
next_to(18,7,19,7).
angle(22,13,6,12,6).
angle(56,13,6,12,7).
angle(63,19,8,12,8).
angle(63,19,8,12,9).
angle(60,19,8,12,10).
angle(55,19,8,12,11).
angle(104,13,6,13,7).
angle(90,19,8,13,8).
angle(81,19,8,13,9).
angle(72,19,8,13,10).
angle(63,19,8,13,11).
angle(158,19,8,14,6).
angle(153,13,6,14,7).
angle(117,19,8,14,8).
angle(97,19,8,14,9).
angle(82,19,8,14,10).
angle(70,19,8,14,11).
angle(153,19,8,15,6).
angle(180,13,6,15,7).
angle(135,19,8,15,8).
angle(110,19,8,15,9).
angle(90,19,8,15,10).
angle(75,19,8,15,11).
angle(146,19,8,16,6).
angle(180,19,8,16,7).
angle(146,19,8,16,8).
angle(117,19,8,16,9).
angle(93,19,8,16,10).
angle(76,19,8,16,11).
angle(135,19,8,17,6).
angle(167,19,8,17,7).
angle(135,17,8,17,9).
angle(117,17,8,17,10).
angle(90,17,8,17,11).
angle(117,19,8,18,6).
angle(146,19,8,18,7).
angle(158,19,8,18,8).
angle(162,17,8,18,9).
angle(162,17,8,18,10).
angle(108,17,8,18,11).
angle(90,19,8,19,6).
angle(99,19,8,19,7).
angle(90,13,6,19,8).
angle(117,17,8,19,9).
angle(135,17,8,19,10).
angle(50,19,8,19,11).
holds(in(learner,13,6),0).
holds(ball_in(13,6),0).
holds(has_ball(learner),0).
holds(agent_in(offense2,11,6),0).
holds(agent_in(defense1,17,8),0).
holds(agent_in(defense2,13,6),0).
holds(agent_in(defense3,14,9),0).
holds(agent_in(defense4,14,6),0).
holds(agent_in(defense5,14,10),0).
holds(agent_in(offense2,11,6),1).
holds(agent_in(defense1,17,8),1).
holds(agent_in(defense2,13,6),1).
holds(agent_in(defense3,14,8),1).
holds(agent_in(defense4,13,6),1).
holds(agent_in(defense5,13,10),1).
display
occurs.
