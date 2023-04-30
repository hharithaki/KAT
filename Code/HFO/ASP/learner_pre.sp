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

display
occurs.
