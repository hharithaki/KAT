#const n = 13.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% grid size     : 320 (16x20)       %
% const term    : not included      %
% directions    : 24                %
% in_range size : 4                 %
% cost terms    : all included      %
% next_to terms : not included      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

sorts
#agent = {learner}.
#guard = {guard2, guard3}.
#attacker = {attacker1, attacker2, attacker3}.
#other_agents = #guard + #attacker.
#all_agents = #agent + #other_agents.
#x_value = 0..19.
#y_value = 0..15.
#value = 0..10.
#sum_val = 0..50.
#direction = {north, south, east, west, north_east, north_west, south_east, south_west, north1, south1, east1, west1, north_east1, north_west1, south_east1, south_west1, 
              north2, south2, east2, west2, north_east2, north_west2, south_east2, south_west2}.
#step = 0..n.
#boolean = {true, false}.

#agent_actions = move(#agent, #x_value, #y_value) + rotate(#agent, #direction) + shoot(#agent, #attacker).
#others_actions = agent_move(#other_agents, #x_value, #y_value) + agent_rotate(#other_agents, #direction) + agent_shoot(#other_agents,#all_agents).
#exogenous_actions = exo_move(#agent, #x_value, #y_value) + exo_rotate(#agent, #direction) + exo_shoot(#agent).

#action = #agent_actions + #others_actions + #exogenous_actions.

#inertial_f = in(#agent, #x_value, #y_value) + face(#agent, #direction) + shot(#all_agents).
#defined_f = distance_gur_to_lea(#guard, #x_value, #y_value) + in_range(#agent, #attacker) + reached_fort(#attacker) + agent_in(#other_agents, #x_value, #y_value) + agent_face(#other_agents, #direction) + agent_shot(#other_agents).

#fluents = #inertial_f + #defined_f.

predicates
fort(#x_value, #y_value).
next_to(#x_value, #y_value, #x_value, #y_value).
next_dir(#direction, #direction).
facing(#direction, #x_value, #y_value, #x_value, #y_value).
holds(#fluents, #step).
occurs(#action, #step).

%%history
hpd(#agent_actions, #step).
obs(#fluents, #boolean, #step).
current_step(#step).
unobserved(#exogenous_actions, #step).
diagnosing(#step).

%%planning
success().
something_happened(#step).
goal(#step).
planning(#step).
cost(#agent_actions,#value).
cost_defined(#agent_actions).
total(#sum_val).

rules
% -------------------------- casual laws --------------------------%

% move causes the agent to be in that place
holds(in(R,X,Y),I+1) :- occurs(move(R,X,Y),I).

% rotate causes the agent to be facing a different direction
holds(face(R,D),I+1) :- occurs(rotate(R,D),I).

% shoot causes the agent to be shot
holds(shot(A),I+1) :- occurs(shoot(R,A),I).

% ----------------------- state constraints -----------------------%

% next_to works both ways. If not specified then not next_to each other
next_to(X1,Y1,X2,Y2) :- next_to(X2,Y2,X1,Y1).
-next_to(X1,Y1,X2,Y2) :- not next_to(X1,Y1,X2,Y2). 

% relationship between directions
next_dir(D1,D2) :- next_dir(D2,D1).
-next_dir(D1,D2) :- not next_dir(D1,D2).

% if not defined as facing then they are not facing each other
-facing(D,X1,Y1,X2,Y2) :- not facing(D,X1,Y1,X2,Y2).

% default cost for agent_actions is 0
cost(A,0) :- #agent_actions(A), not cost_defined(A).

% minimum distance between guards is 4 grid cells. Hence moving closer will incur a cost of 2
cost(move(R,X,Y),2) :- holds(agent_in(G,XG,YG),I), #guard(G), XG < X, X-XG <= 4.
cost(move(R,X,Y),2) :- holds(agent_in(G,XG,YG),I), #guard(G), X < XG, XG-X <= 4.

% moving away from attackers incur a cost - temp => attackers does not shoot at the moment
cost(move(R,X,Y),1) :- holds(agent_in(A,XA,YA),I), #attacker(A), holds(in(R,X0,Y0),I), not cost(move(R,X,Y),V), V = 2, X0-XA > X-XA, X0 > XA, X > XA.
cost(move(R,X,Y),1) :- holds(agent_in(A,XA,YA),I), #attacker(A), holds(in(R,X0,Y0),I), not cost(move(R,X,Y),V), V = 2, XA-X0 > XA-X, X0 < XA, X < XA.
cost(move(R,X,Y),1) :- holds(agent_in(A,XA,YA),I), #attacker(A), holds(in(R,X0,Y0),I), not cost(move(R,X,Y),V), V = 2, Y0-YA > Y-YA, Y0 > YA, Y > YA.
cost(move(R,X,Y),1) :- holds(agent_in(A,XA,YA),I), #attacker(A), holds(in(R,X0,Y0),I), not cost(move(R,X,Y),V), V = 2, YA-Y0 > YA-Y, Y0 < YA, Y < YA.

cost_defined(A) :- cost(A,V), V != 0. 

% an agent cannot be in two places at the same time.
-holds(in(R,X1,Y1),I) :- holds(in(R,X2,Y2),I), X1 != X2, Y1 != Y2.
-holds(in(R,X1,Y1),I) :- holds(in(R,X2,Y2),I), X1 != X2, Y1 = Y2.
-holds(in(R,X1,Y1),I) :- holds(in(R,X2,Y2),I), X1 = X2, Y1 != Y2.

% agent cannot face two direction at once
-holds(face(R,D),I) :- holds(face(R,D1),I), D != D1.

%% updating defined fleuents %%
holds(agent_shot(A),I) :- holds(shot(A),I).
holds(reached_fort(A),I) :- holds(agent_in(A,X,Y),I), fort(X,Y), #attacker(A).
holds(distance_gur_to_lea(G,X-XG,Y-YG),I) :- holds(agent_in(G,XG,YG),I), holds(in(learner,X,Y),I), #guard(G), X >= XG, Y >= YG.
holds(distance_gur_to_lea(G,XG-X,YG-Y),I) :- holds(agent_in(G,XG,YG),I), holds(in(learner,X,Y),I), #guard(G), X <= XG, Y <= YG.
holds(distance_gur_to_lea(G,X-XG,YG-Y),I) :- holds(agent_in(G,XG,YG),I), holds(in(learner,X,Y),I), #guard(G), X > XG, Y < YG.
holds(distance_gur_to_lea(G,XG-X,Y-YG),I) :- holds(agent_in(G,XG,YG),I), holds(in(learner,X,Y),I), #guard(G), X < XG, Y > YG.

% the calculatio should be 4. But in reality robot does not move the distance expected
holds(in_range(L,A),I) :- holds(agent_in(A,XA,YA),I), holds(in(L,XL,YL),I), holds(face(L,south),I), facing(south,XL,YL,XA,YA), #attacker(A), YL-YA <= 8, XL-3 <= XA, XA <= XL+3.
holds(in_range(L,A),I) :- holds(agent_in(A,XA,YA),I), holds(in(L,XL,YL),I), holds(face(L,north),I), facing(north,XL,YL,XA,YA), #attacker(A), YA-YL <= 8, XL-3 <= XA, XA <= XL+3.
holds(in_range(L,A),I) :- holds(agent_in(A,XA,YA),I), holds(in(L,XL,YL),I), holds(face(L,west),I), facing(west,XL,YL,XA,YA), #attacker(A), XL-XA <= 8, YL-3 <= YA, YA <= YL+3.
holds(in_range(L,A),I) :- holds(agent_in(A,XA,YA),I), holds(in(L,XL,YL),I), holds(face(L,east),I), facing(east,XL,YL,XA,YA), #attacker(A), XA-XL <= 8, YL-3 <= YA, YA <= YL+3.

holds(in_range(L,A),I) :- holds(agent_in(A,XA,YA),I), holds(in(L,XL,YL),I), holds(face(L,south_west),I), facing(south_west,XL,YL,XA,YA), #attacker(A), XL-XA <= 4, YL-YA <= 4.
holds(in_range(L,A),I) :- holds(agent_in(A,XA,YA),I), holds(in(L,XL,YL),I), holds(face(L,north_west),I), facing(north_west,XL,YL,XA,YA), #attacker(A), XL-XA <= 4, YA-YL <= 4.
holds(in_range(L,A),I) :- holds(agent_in(A,XA,YA),I), holds(in(L,XL,YL),I), holds(face(L,south_east),I), facing(south_east,XL,YL,XA,YA), #attacker(A), XA-XL <= 4, YL-YA <= 4.
holds(in_range(L,A),I) :- holds(agent_in(A,XA,YA),I), holds(in(L,XL,YL),I), holds(face(L,north_east),I), facing(north_east,XL,YL,XA,YA), #attacker(A), XA-XL <= 4, YA-YL <= 4.

holds(agent_in(A,X,Y),I+1) :- holds(agent_in(A,X,Y),I), not holds(agent_in(A,X1,Y1),I+1), I >= 1, X1!=X,Y1!=Y.
holds(agent_face(A,D),I+1) :- holds(agent_face(A,D),I), not holds(agent_face(A,D1),I+1), I >= 1, D1!=D.
holds(shot(A),I+1) :- holds(shot(A),I).

% extend to T shape - in reality 7 should have 3! but as agent also moves 3 is not practical
holds(in_range(L,A),I) :- holds(agent_in(A,XA,YA),I), holds(in(L,XL,YL),I), holds(face(L,south),I), #attacker(A), YL-YA <= 7, YL-YA >= 6, XL-3 <= XA, XA <= XL+3.
holds(in_range(L,A),I) :- holds(agent_in(A,XA,YA),I), holds(in(L,XL,YL),I), holds(face(L,north),I), #attacker(A), YA-YL <= 7, YA-YL >= 6, XL-3 <= XA, XA <= XL+3.
holds(in_range(L,A),I) :- holds(agent_in(A,XA,YA),I), holds(in(L,XL,YL),I), holds(face(L,west),I), #attacker(A), XL-XA <= 7, XL-XA >= 6, YL-3 <= YA, YA <= YL+3.
holds(in_range(L,A),I) :- holds(agent_in(A,XA,YA),I), holds(in(L,XL,YL),I), holds(face(L,east),I), #attacker(A), XA-XL <= 7, XA-XL >= 6, YL-3 <= YA, YA <= YL+3.

holds(in_range(L,A),I) :- holds(agent_in(A,XA,YA),I), holds(in(L,XL,YL),I), holds(face(L,south),I), #attacker(A), YL-YA <= 6, YL-YA >= 5, XL-2 <= XA, XA <= XL+2.
holds(in_range(L,A),I) :- holds(agent_in(A,XA,YA),I), holds(in(L,XL,YL),I), holds(face(L,south),I), #attacker(A), YL-YA <= 5, YL-YA >= 4, XL-2 <= XA, XA <= XL+2.
%holds(in_range(L,A),I) :- holds(agent_in(A,XA,YA),I), holds(in(L,XL,YL),I), holds(face(L,south),I), #attacker(A), YL-YA <= 4, YL-YA >= 3, XL-1 <= XA, XA <= XL+1.
%holds(in_range(L,A),I) :- holds(agent_in(A,XA,YA),I), holds(in(L,XL,YL),I), holds(face(L,south),I), #attacker(A), YL-YA <= 3, YL-YA >= 2, XL-1 <= XA, XA <= XL+1.

% ------------------------ inertial axioms ------------------------%

holds(F,I+1) :- #inertial_f(F), holds(F,I), not -holds(F,I+1).
-holds(F,I+1) :- #inertial_f(F), -holds(F,I), not holds(F,I+1).

% ------------------------------ CWA ------------------------------%

-occurs(A,I) :- not occurs(A,I).
-holds(F,I) :- #defined_f(F), not holds(F,I).

% -------------------- executability conditions -------------------%

% impossible to move to a location it is already in
-occurs(move(R,X,Y),I) :- holds(in(R,X,Y),I). 

% impossible to move between locations which are not next to each other
-occurs(move(R,X1,Y1),I) :- holds(in(R,X2,Y2),I), -next_to(X1,Y1,X2,Y2).

% impossible to rotate to a direction which is not next to its current(facing) direction
-occurs(rotate(R,D1),I) :- holds(face(R,D2),I), -next_dir(D1,D2).

% impossible to shoot an agent who has already been shot
-occurs(shoot(R,A),I) :- holds(agent_shot(A),I). 

% impossible to shoot another agent if that agent is not in the shooting range
-occurs(shoot(R,A),I) :- -holds(in_range(R,A),I).

% ---------------------------- history ----------------------------%
% guarantees that agent takes all fluents in the system into consideration
holds(F,0) | -holds(F,0) :- #inertial_f(F).

% record all the actions that happened
occurs(A,I) :- hpd(A,I), current_step(I1), I < I1.

% reality check axioms. gurantees that the agents expectations agrees with its observations
:- current_step(I1), I <= I1, obs(F,true,I), -holds(F,I).
:- current_step(I1), I <= I1, obs(F,false,I), holds(F,I).

% record unobserved exogenous action occurrences
occurs(A,I) :- unobserved(A,I), #exogenous_actions(A).

% generate minimal explanations.
unobserved(A,I0) :+ current_step(I1), diagnosing(I1), I0 < I1, not hpd(A,I0), #exogenous_actions(A).

% --------------------------- planning ---------------------------%

% to achieve success the system should satisfies the goal. Failure is not acceptable
success :- goal(I), I <= n.
:- not success, current_step(I0), planning(I0).

% consider the occurrence of exogenous actions when they are absolutely necessary for resolving a conflict
occurs(A,I) :+ #agent_actions(A), #step(I), current_step(I0), planning(I0), I0 < I.

% agent can not execute parallel actions
-occurs(A1,I) :- occurs(A2,I), A1 != A2, #agent_actions(A1), #agent_actions(A2).

% an action should occur at each time step until the goal is achieved
something_happened(I1) :- current_step(I0), planning(I0), I0 <= I1, occurs(A,I1), #agent_actions(A).
:- not something_happened(I), something_happened(I+1), I0 < I, current_step(I0), planning(I0).

total(S) :- S = #sum{C, A:occurs(A,I), cost(A,C)}.
#minimize {V@2, V:total(V)}.

%%%--------------------------------------------------------------%%%

planning(I) | diagnosing(I) :- current_step(I).

%define relation

fort(4,7).
fort(5,7).

planning(0).
current_step(0).

% --------------- %

next_dir(north,north_east1).
next_dir(north_east1,north_east2).
next_dir(north_east2,north_east).
next_dir(north_east,east1).
next_dir(east1,east2).
next_dir(east2,east).
next_dir(east,south_east1).
next_dir(south_east1,south_east2).
next_dir(south_east2,south_east).
next_dir(south_east,south1).
next_dir(south1,south2).
next_dir(south2,south).
next_dir(south,south_west1).
next_dir(south_west1,south_west2).
next_dir(south_west2,south_west).
next_dir(south_west,west1).
next_dir(west1,west2).
next_dir(west2,west).
next_dir(west,north_west1).
next_dir(north_west1,north_west2).
next_dir(north_west2,north_west).
next_dir(north_west,north1).
next_dir(north1,north2).
next_dir(north2,north).
next_to(4,7,4,8).
next_to(4,7,5,7).
next_to(5,7,5,8).
next_to(5,7,6,7).
next_to(6,7,6,8).
next_to(6,7,7,7).
next_to(7,7,7,8).
next_to(7,7,8,7).
next_to(8,7,8,8).
next_to(8,7,9,7).
next_to(9,7,9,8).
next_to(9,7,10,7).
next_to(10,7,10,8).
next_to(4,8,4,9).
next_to(4,8,5,8).
next_to(5,8,5,9).
next_to(5,8,6,8).
next_to(6,8,6,9).
next_to(6,8,7,8).
next_to(7,8,7,9).
next_to(7,8,8,8).
next_to(8,8,8,9).
next_to(8,8,9,8).
next_to(9,8,9,9).
next_to(9,8,10,8).
next_to(10,8,10,9).
next_to(4,9,4,10).
next_to(4,9,5,9).
next_to(5,9,5,10).
next_to(5,9,6,9).
next_to(6,9,6,10).
next_to(6,9,7,9).
next_to(7,9,7,10).
next_to(7,9,8,9).
next_to(8,9,8,10).
next_to(8,9,9,9).
next_to(9,9,9,10).
next_to(9,9,10,9).
next_to(10,9,10,10).
next_to(4,10,4,11).
next_to(4,10,5,10).
next_to(5,10,5,11).
next_to(5,10,6,10).
next_to(6,10,6,11).
next_to(6,10,7,10).
next_to(7,10,7,11).
next_to(7,10,8,10).
next_to(8,10,8,11).
next_to(8,10,9,10).
next_to(9,10,9,11).
next_to(9,10,10,10).
next_to(10,10,10,11).
next_to(4,11,4,12).
next_to(4,11,5,11).
next_to(5,11,5,12).
next_to(5,11,6,11).
next_to(6,11,6,12).
next_to(6,11,7,11).
next_to(7,11,7,12).
next_to(7,11,8,11).
next_to(8,11,8,12).
next_to(8,11,9,11).
next_to(9,11,9,12).
next_to(9,11,10,11).
next_to(10,11,10,12).
next_to(4,12,4,13).
next_to(4,12,5,12).
next_to(5,12,5,13).
next_to(5,12,6,12).
next_to(6,12,6,13).
next_to(6,12,7,12).
next_to(7,12,7,13).
next_to(7,12,8,12).
next_to(8,12,8,13).
next_to(8,12,9,12).
next_to(9,12,9,13).
next_to(9,12,10,12).
next_to(10,12,10,13).
next_to(4,13,4,14).
next_to(4,13,5,13).
next_to(5,13,5,14).
next_to(5,13,6,13).
next_to(6,13,6,14).
next_to(6,13,7,13).
next_to(7,13,7,14).
next_to(7,13,8,13).
next_to(8,13,8,14).
next_to(8,13,9,13).
next_to(9,13,9,14).
next_to(9,13,10,13).
next_to(10,13,10,14).
next_to(4,14,5,14).
next_to(5,14,6,14).
next_to(6,14,7,14).
next_to(7,14,8,14).
next_to(8,14,9,14).
next_to(9,14,10,14).
facing(north,4,7,4,8).
facing(north,4,7,4,9).
facing(north,4,7,4,10).
facing(north,4,7,4,11).
facing(north,4,7,4,12).
facing(north,4,7,4,13).
facing(north,4,7,4,14).
facing(north,4,8,4,9).
facing(north,4,8,4,10).
facing(north,4,8,4,11).
facing(north,4,8,4,12).
facing(north,4,8,4,13).
facing(north,4,8,4,14).
facing(north,4,9,4,10).
facing(north,4,9,4,11).
facing(north,4,9,4,12).
facing(north,4,9,4,13).
facing(north,4,9,4,14).
facing(north,4,10,4,11).
facing(north,4,10,4,12).
facing(north,4,10,4,13).
facing(north,4,10,4,14).
facing(north,4,11,4,12).
facing(north,4,11,4,13).
facing(north,4,11,4,14).
facing(north,4,12,4,13).
facing(north,4,12,4,14).
facing(north,4,13,4,14).
facing(north,5,7,5,8).
facing(north,5,7,5,9).
facing(north,5,7,5,10).
facing(north,5,7,5,11).
facing(north,5,7,5,12).
facing(north,5,7,5,13).
facing(north,5,7,5,14).
facing(north,5,8,5,9).
facing(north,5,8,5,10).
facing(north,5,8,5,11).
facing(north,5,8,5,12).
facing(north,5,8,5,13).
facing(north,5,8,5,14).
facing(north,5,9,5,10).
facing(north,5,9,5,11).
facing(north,5,9,5,12).
facing(north,5,9,5,13).
facing(north,5,9,5,14).
facing(north,5,10,5,11).
facing(north,5,10,5,12).
facing(north,5,10,5,13).
facing(north,5,10,5,14).
facing(north,5,11,5,12).
facing(north,5,11,5,13).
facing(north,5,11,5,14).
facing(north,5,12,5,13).
facing(north,5,12,5,14).
facing(north,5,13,5,14).
facing(north,6,7,6,8).
facing(north,6,7,6,9).
facing(north,6,7,6,10).
facing(north,6,7,6,11).
facing(north,6,7,6,12).
facing(north,6,7,6,13).
facing(north,6,7,6,14).
facing(north,6,8,6,9).
facing(north,6,8,6,10).
facing(north,6,8,6,11).
facing(north,6,8,6,12).
facing(north,6,8,6,13).
facing(north,6,8,6,14).
facing(north,6,9,6,10).
facing(north,6,9,6,11).
facing(north,6,9,6,12).
facing(north,6,9,6,13).
facing(north,6,9,6,14).
facing(north,6,10,6,11).
facing(north,6,10,6,12).
facing(north,6,10,6,13).
facing(north,6,10,6,14).
facing(north,6,11,6,12).
facing(north,6,11,6,13).
facing(north,6,11,6,14).
facing(north,6,12,6,13).
facing(north,6,12,6,14).
facing(north,6,13,6,14).
facing(north,7,7,7,8).
facing(north,7,7,7,9).
facing(north,7,7,7,10).
facing(north,7,7,7,11).
facing(north,7,7,7,12).
facing(north,7,7,7,13).
facing(north,7,7,7,14).
facing(north,7,8,7,9).
facing(north,7,8,7,10).
facing(north,7,8,7,11).
facing(north,7,8,7,12).
facing(north,7,8,7,13).
facing(north,7,8,7,14).
facing(north,7,9,7,10).
facing(north,7,9,7,11).
facing(north,7,9,7,12).
facing(north,7,9,7,13).
facing(north,7,9,7,14).
facing(north,7,10,7,11).
facing(north,7,10,7,12).
facing(north,7,10,7,13).
facing(north,7,10,7,14).
facing(north,7,11,7,12).
facing(north,7,11,7,13).
facing(north,7,11,7,14).
facing(north,7,12,7,13).
facing(north,7,12,7,14).
facing(north,7,13,7,14).
facing(north,8,7,8,8).
facing(north,8,7,8,9).
facing(north,8,7,8,10).
facing(north,8,7,8,11).
facing(north,8,7,8,12).
facing(north,8,7,8,13).
facing(north,8,7,8,14).
facing(north,8,8,8,9).
facing(north,8,8,8,10).
facing(north,8,8,8,11).
facing(north,8,8,8,12).
facing(north,8,8,8,13).
facing(north,8,8,8,14).
facing(north,8,9,8,10).
facing(north,8,9,8,11).
facing(north,8,9,8,12).
facing(north,8,9,8,13).
facing(north,8,9,8,14).
facing(north,8,10,8,11).
facing(north,8,10,8,12).
facing(north,8,10,8,13).
facing(north,8,10,8,14).
facing(north,8,11,8,12).
facing(north,8,11,8,13).
facing(north,8,11,8,14).
facing(north,8,12,8,13).
facing(north,8,12,8,14).
facing(north,8,13,8,14).
facing(north,9,7,9,8).
facing(north,9,7,9,9).
facing(north,9,7,9,10).
facing(north,9,7,9,11).
facing(north,9,7,9,12).
facing(north,9,7,9,13).
facing(north,9,7,9,14).
facing(north,9,8,9,9).
facing(north,9,8,9,10).
facing(north,9,8,9,11).
facing(north,9,8,9,12).
facing(north,9,8,9,13).
facing(north,9,8,9,14).
facing(north,9,9,9,10).
facing(north,9,9,9,11).
facing(north,9,9,9,12).
facing(north,9,9,9,13).
facing(north,9,9,9,14).
facing(north,9,10,9,11).
facing(north,9,10,9,12).
facing(north,9,10,9,13).
facing(north,9,10,9,14).
facing(north,9,11,9,12).
facing(north,9,11,9,13).
facing(north,9,11,9,14).
facing(north,9,12,9,13).
facing(north,9,12,9,14).
facing(north,9,13,9,14).
facing(north,10,7,10,8).
facing(north,10,7,10,9).
facing(north,10,7,10,10).
facing(north,10,7,10,11).
facing(north,10,7,10,12).
facing(north,10,7,10,13).
facing(north,10,7,10,14).
facing(north,10,8,10,9).
facing(north,10,8,10,10).
facing(north,10,8,10,11).
facing(north,10,8,10,12).
facing(north,10,8,10,13).
facing(north,10,8,10,14).
facing(north,10,9,10,10).
facing(north,10,9,10,11).
facing(north,10,9,10,12).
facing(north,10,9,10,13).
facing(north,10,9,10,14).
facing(north,10,10,10,11).
facing(north,10,10,10,12).
facing(north,10,10,10,13).
facing(north,10,10,10,14).
facing(north,10,11,10,12).
facing(north,10,11,10,13).
facing(north,10,11,10,14).
facing(north,10,12,10,13).
facing(north,10,12,10,14).
facing(north,10,13,10,14).
facing(south,4,14,4,13).
facing(south,4,14,4,12).
facing(south,4,14,4,11).
facing(south,4,14,4,10).
facing(south,4,14,4,9).
facing(south,4,14,4,8).
facing(south,4,14,4,7).
facing(south,4,13,4,12).
facing(south,4,13,4,11).
facing(south,4,13,4,10).
facing(south,4,13,4,9).
facing(south,4,13,4,8).
facing(south,4,13,4,7).
facing(south,4,12,4,11).
facing(south,4,12,4,10).
facing(south,4,12,4,9).
facing(south,4,12,4,8).
facing(south,4,12,4,7).
facing(south,4,11,4,10).
facing(south,4,11,4,9).
facing(south,4,11,4,8).
facing(south,4,11,4,7).
facing(south,4,10,4,9).
facing(south,4,10,4,8).
facing(south,4,10,4,7).
facing(south,4,9,4,8).
facing(south,4,9,4,7).
facing(south,4,8,4,7).
facing(south,5,14,5,13).
facing(south,5,14,5,12).
facing(south,5,14,5,11).
facing(south,5,14,5,10).
facing(south,5,14,5,9).
facing(south,5,14,5,8).
facing(south,5,14,5,7).
facing(south,5,13,5,12).
facing(south,5,13,5,11).
facing(south,5,13,5,10).
facing(south,5,13,5,9).
facing(south,5,13,5,8).
facing(south,5,13,5,7).
facing(south,5,12,5,11).
facing(south,5,12,5,10).
facing(south,5,12,5,9).
facing(south,5,12,5,8).
facing(south,5,12,5,7).
facing(south,5,11,5,10).
facing(south,5,11,5,9).
facing(south,5,11,5,8).
facing(south,5,11,5,7).
facing(south,5,10,5,9).
facing(south,5,10,5,8).
facing(south,5,10,5,7).
facing(south,5,9,5,8).
facing(south,5,9,5,7).
facing(south,5,8,5,7).
facing(south,6,14,6,13).
facing(south,6,14,6,12).
facing(south,6,14,6,11).
facing(south,6,14,6,10).
facing(south,6,14,6,9).
facing(south,6,14,6,8).
facing(south,6,14,6,7).
facing(south,6,13,6,12).
facing(south,6,13,6,11).
facing(south,6,13,6,10).
facing(south,6,13,6,9).
facing(south,6,13,6,8).
facing(south,6,13,6,7).
facing(south,6,12,6,11).
facing(south,6,12,6,10).
facing(south,6,12,6,9).
facing(south,6,12,6,8).
facing(south,6,12,6,7).
facing(south,6,11,6,10).
facing(south,6,11,6,9).
facing(south,6,11,6,8).
facing(south,6,11,6,7).
facing(south,6,10,6,9).
facing(south,6,10,6,8).
facing(south,6,10,6,7).
facing(south,6,9,6,8).
facing(south,6,9,6,7).
facing(south,6,8,6,7).
facing(south,7,14,7,13).
facing(south,7,14,7,12).
facing(south,7,14,7,11).
facing(south,7,14,7,10).
facing(south,7,14,7,9).
facing(south,7,14,7,8).
facing(south,7,14,7,7).
facing(south,7,13,7,12).
facing(south,7,13,7,11).
facing(south,7,13,7,10).
facing(south,7,13,7,9).
facing(south,7,13,7,8).
facing(south,7,13,7,7).
facing(south,7,12,7,11).
facing(south,7,12,7,10).
facing(south,7,12,7,9).
facing(south,7,12,7,8).
facing(south,7,12,7,7).
facing(south,7,11,7,10).
facing(south,7,11,7,9).
facing(south,7,11,7,8).
facing(south,7,11,7,7).
facing(south,7,10,7,9).
facing(south,7,10,7,8).
facing(south,7,10,7,7).
facing(south,7,9,7,8).
facing(south,7,9,7,7).
facing(south,7,8,7,7).
facing(south,8,14,8,13).
facing(south,8,14,8,12).
facing(south,8,14,8,11).
facing(south,8,14,8,10).
facing(south,8,14,8,9).
facing(south,8,14,8,8).
facing(south,8,14,8,7).
facing(south,8,13,8,12).
facing(south,8,13,8,11).
facing(south,8,13,8,10).
facing(south,8,13,8,9).
facing(south,8,13,8,8).
facing(south,8,13,8,7).
facing(south,8,12,8,11).
facing(south,8,12,8,10).
facing(south,8,12,8,9).
facing(south,8,12,8,8).
facing(south,8,12,8,7).
facing(south,8,11,8,10).
facing(south,8,11,8,9).
facing(south,8,11,8,8).
facing(south,8,11,8,7).
facing(south,8,10,8,9).
facing(south,8,10,8,8).
facing(south,8,10,8,7).
facing(south,8,9,8,8).
facing(south,8,9,8,7).
facing(south,8,8,8,7).
facing(south,9,14,9,13).
facing(south,9,14,9,12).
facing(south,9,14,9,11).
facing(south,9,14,9,10).
facing(south,9,14,9,9).
facing(south,9,14,9,8).
facing(south,9,14,9,7).
facing(south,9,13,9,12).
facing(south,9,13,9,11).
facing(south,9,13,9,10).
facing(south,9,13,9,9).
facing(south,9,13,9,8).
facing(south,9,13,9,7).
facing(south,9,12,9,11).
facing(south,9,12,9,10).
facing(south,9,12,9,9).
facing(south,9,12,9,8).
facing(south,9,12,9,7).
facing(south,9,11,9,10).
facing(south,9,11,9,9).
facing(south,9,11,9,8).
facing(south,9,11,9,7).
facing(south,9,10,9,9).
facing(south,9,10,9,8).
facing(south,9,10,9,7).
facing(south,9,9,9,8).
facing(south,9,9,9,7).
facing(south,9,8,9,7).
facing(south,10,14,10,13).
facing(south,10,14,10,12).
facing(south,10,14,10,11).
facing(south,10,14,10,10).
facing(south,10,14,10,9).
facing(south,10,14,10,8).
facing(south,10,14,10,7).
facing(south,10,13,10,12).
facing(south,10,13,10,11).
facing(south,10,13,10,10).
facing(south,10,13,10,9).
facing(south,10,13,10,8).
facing(south,10,13,10,7).
facing(south,10,12,10,11).
facing(south,10,12,10,10).
facing(south,10,12,10,9).
facing(south,10,12,10,8).
facing(south,10,12,10,7).
facing(south,10,11,10,10).
facing(south,10,11,10,9).
facing(south,10,11,10,8).
facing(south,10,11,10,7).
facing(south,10,10,10,9).
facing(south,10,10,10,8).
facing(south,10,10,10,7).
facing(south,10,9,10,8).
facing(south,10,9,10,7).
facing(south,10,8,10,7).
facing(east,4,7,5,7).
facing(east,4,7,6,7).
facing(east,4,7,7,7).
facing(east,4,7,8,7).
facing(east,4,7,9,7).
facing(east,4,7,10,7).
facing(east,5,7,6,7).
facing(east,5,7,7,7).
facing(east,5,7,8,7).
facing(east,5,7,9,7).
facing(east,5,7,10,7).
facing(east,6,7,7,7).
facing(east,6,7,8,7).
facing(east,6,7,9,7).
facing(east,6,7,10,7).
facing(east,7,7,8,7).
facing(east,7,7,9,7).
facing(east,7,7,10,7).
facing(east,8,7,9,7).
facing(east,8,7,10,7).
facing(east,9,7,10,7).
facing(east,4,8,5,8).
facing(east,4,8,6,8).
facing(east,4,8,7,8).
facing(east,4,8,8,8).
facing(east,4,8,9,8).
facing(east,4,8,10,8).
facing(east,5,8,6,8).
facing(east,5,8,7,8).
facing(east,5,8,8,8).
facing(east,5,8,9,8).
facing(east,5,8,10,8).
facing(east,6,8,7,8).
facing(east,6,8,8,8).
facing(east,6,8,9,8).
facing(east,6,8,10,8).
facing(east,7,8,8,8).
facing(east,7,8,9,8).
facing(east,7,8,10,8).
facing(east,8,8,9,8).
facing(east,8,8,10,8).
facing(east,9,8,10,8).
facing(east,4,9,5,9).
facing(east,4,9,6,9).
facing(east,4,9,7,9).
facing(east,4,9,8,9).
facing(east,4,9,9,9).
facing(east,4,9,10,9).
facing(east,5,9,6,9).
facing(east,5,9,7,9).
facing(east,5,9,8,9).
facing(east,5,9,9,9).
facing(east,5,9,10,9).
facing(east,6,9,7,9).
facing(east,6,9,8,9).
facing(east,6,9,9,9).
facing(east,6,9,10,9).
facing(east,7,9,8,9).
facing(east,7,9,9,9).
facing(east,7,9,10,9).
facing(east,8,9,9,9).
facing(east,8,9,10,9).
facing(east,9,9,10,9).
facing(east,4,10,5,10).
facing(east,4,10,6,10).
facing(east,4,10,7,10).
facing(east,4,10,8,10).
facing(east,4,10,9,10).
facing(east,4,10,10,10).
facing(east,5,10,6,10).
facing(east,5,10,7,10).
facing(east,5,10,8,10).
facing(east,5,10,9,10).
facing(east,5,10,10,10).
facing(east,6,10,7,10).
facing(east,6,10,8,10).
facing(east,6,10,9,10).
facing(east,6,10,10,10).
facing(east,7,10,8,10).
facing(east,7,10,9,10).
facing(east,7,10,10,10).
facing(east,8,10,9,10).
facing(east,8,10,10,10).
facing(east,9,10,10,10).
facing(east,4,11,5,11).
facing(east,4,11,6,11).
facing(east,4,11,7,11).
facing(east,4,11,8,11).
facing(east,4,11,9,11).
facing(east,4,11,10,11).
facing(east,5,11,6,11).
facing(east,5,11,7,11).
facing(east,5,11,8,11).
facing(east,5,11,9,11).
facing(east,5,11,10,11).
facing(east,6,11,7,11).
facing(east,6,11,8,11).
facing(east,6,11,9,11).
facing(east,6,11,10,11).
facing(east,7,11,8,11).
facing(east,7,11,9,11).
facing(east,7,11,10,11).
facing(east,8,11,9,11).
facing(east,8,11,10,11).
facing(east,9,11,10,11).
facing(east,4,12,5,12).
facing(east,4,12,6,12).
facing(east,4,12,7,12).
facing(east,4,12,8,12).
facing(east,4,12,9,12).
facing(east,4,12,10,12).
facing(east,5,12,6,12).
facing(east,5,12,7,12).
facing(east,5,12,8,12).
facing(east,5,12,9,12).
facing(east,5,12,10,12).
facing(east,6,12,7,12).
facing(east,6,12,8,12).
facing(east,6,12,9,12).
facing(east,6,12,10,12).
facing(east,7,12,8,12).
facing(east,7,12,9,12).
facing(east,7,12,10,12).
facing(east,8,12,9,12).
facing(east,8,12,10,12).
facing(east,9,12,10,12).
facing(east,4,13,5,13).
facing(east,4,13,6,13).
facing(east,4,13,7,13).
facing(east,4,13,8,13).
facing(east,4,13,9,13).
facing(east,4,13,10,13).
facing(east,5,13,6,13).
facing(east,5,13,7,13).
facing(east,5,13,8,13).
facing(east,5,13,9,13).
facing(east,5,13,10,13).
facing(east,6,13,7,13).
facing(east,6,13,8,13).
facing(east,6,13,9,13).
facing(east,6,13,10,13).
facing(east,7,13,8,13).
facing(east,7,13,9,13).
facing(east,7,13,10,13).
facing(east,8,13,9,13).
facing(east,8,13,10,13).
facing(east,9,13,10,13).
facing(east,4,14,5,14).
facing(east,4,14,6,14).
facing(east,4,14,7,14).
facing(east,4,14,8,14).
facing(east,4,14,9,14).
facing(east,4,14,10,14).
facing(east,5,14,6,14).
facing(east,5,14,7,14).
facing(east,5,14,8,14).
facing(east,5,14,9,14).
facing(east,5,14,10,14).
facing(east,6,14,7,14).
facing(east,6,14,8,14).
facing(east,6,14,9,14).
facing(east,6,14,10,14).
facing(east,7,14,8,14).
facing(east,7,14,9,14).
facing(east,7,14,10,14).
facing(east,8,14,9,14).
facing(east,8,14,10,14).
facing(east,9,14,10,14).
facing(west,10,14,10,14).
facing(west,10,14,9,14).
facing(west,10,14,8,14).
facing(west,10,14,7,14).
facing(west,10,14,6,14).
facing(west,10,14,5,14).
facing(west,10,14,4,14).
facing(west,9,14,8,14).
facing(west,9,14,7,14).
facing(west,9,14,6,14).
facing(west,9,14,5,14).
facing(west,9,14,4,14).
facing(west,8,14,7,14).
facing(west,8,14,6,14).
facing(west,8,14,5,14).
facing(west,8,14,4,14).
facing(west,7,14,6,14).
facing(west,7,14,5,14).
facing(west,7,14,4,14).
facing(west,6,14,5,14).
facing(west,6,14,4,14).
facing(west,5,14,4,14).
facing(west,10,13,10,13).
facing(west,10,13,9,13).
facing(west,10,13,8,13).
facing(west,10,13,7,13).
facing(west,10,13,6,13).
facing(west,10,13,5,13).
facing(west,10,13,4,13).
facing(west,9,13,8,13).
facing(west,9,13,7,13).
facing(west,9,13,6,13).
facing(west,9,13,5,13).
facing(west,9,13,4,13).
facing(west,8,13,7,13).
facing(west,8,13,6,13).
facing(west,8,13,5,13).
facing(west,8,13,4,13).
facing(west,7,13,6,13).
facing(west,7,13,5,13).
facing(west,7,13,4,13).
facing(west,6,13,5,13).
facing(west,6,13,4,13).
facing(west,5,13,4,13).
facing(west,10,12,10,12).
facing(west,10,12,9,12).
facing(west,10,12,8,12).
facing(west,10,12,7,12).
facing(west,10,12,6,12).
facing(west,10,12,5,12).
facing(west,10,12,4,12).
facing(west,9,12,8,12).
facing(west,9,12,7,12).
facing(west,9,12,6,12).
facing(west,9,12,5,12).
facing(west,9,12,4,12).
facing(west,8,12,7,12).
facing(west,8,12,6,12).
facing(west,8,12,5,12).
facing(west,8,12,4,12).
facing(west,7,12,6,12).
facing(west,7,12,5,12).
facing(west,7,12,4,12).
facing(west,6,12,5,12).
facing(west,6,12,4,12).
facing(west,5,12,4,12).
facing(west,10,11,10,11).
facing(west,10,11,9,11).
facing(west,10,11,8,11).
facing(west,10,11,7,11).
facing(west,10,11,6,11).
facing(west,10,11,5,11).
facing(west,10,11,4,11).
facing(west,9,11,8,11).
facing(west,9,11,7,11).
facing(west,9,11,6,11).
facing(west,9,11,5,11).
facing(west,9,11,4,11).
facing(west,8,11,7,11).
facing(west,8,11,6,11).
facing(west,8,11,5,11).
facing(west,8,11,4,11).
facing(west,7,11,6,11).
facing(west,7,11,5,11).
facing(west,7,11,4,11).
facing(west,6,11,5,11).
facing(west,6,11,4,11).
facing(west,5,11,4,11).
facing(west,10,10,10,10).
facing(west,10,10,9,10).
facing(west,10,10,8,10).
facing(west,10,10,7,10).
facing(west,10,10,6,10).
facing(west,10,10,5,10).
facing(west,10,10,4,10).
facing(west,9,10,8,10).
facing(west,9,10,7,10).
facing(west,9,10,6,10).
facing(west,9,10,5,10).
facing(west,9,10,4,10).
facing(west,8,10,7,10).
facing(west,8,10,6,10).
facing(west,8,10,5,10).
facing(west,8,10,4,10).
facing(west,7,10,6,10).
facing(west,7,10,5,10).
facing(west,7,10,4,10).
facing(west,6,10,5,10).
facing(west,6,10,4,10).
facing(west,5,10,4,10).
facing(west,10,9,10,9).
facing(west,10,9,9,9).
facing(west,10,9,8,9).
facing(west,10,9,7,9).
facing(west,10,9,6,9).
facing(west,10,9,5,9).
facing(west,10,9,4,9).
facing(west,9,9,8,9).
facing(west,9,9,7,9).
facing(west,9,9,6,9).
facing(west,9,9,5,9).
facing(west,9,9,4,9).
facing(west,8,9,7,9).
facing(west,8,9,6,9).
facing(west,8,9,5,9).
facing(west,8,9,4,9).
facing(west,7,9,6,9).
facing(west,7,9,5,9).
facing(west,7,9,4,9).
facing(west,6,9,5,9).
facing(west,6,9,4,9).
facing(west,5,9,4,9).
facing(west,10,8,10,8).
facing(west,10,8,9,8).
facing(west,10,8,8,8).
facing(west,10,8,7,8).
facing(west,10,8,6,8).
facing(west,10,8,5,8).
facing(west,10,8,4,8).
facing(west,9,8,8,8).
facing(west,9,8,7,8).
facing(west,9,8,6,8).
facing(west,9,8,5,8).
facing(west,9,8,4,8).
facing(west,8,8,7,8).
facing(west,8,8,6,8).
facing(west,8,8,5,8).
facing(west,8,8,4,8).
facing(west,7,8,6,8).
facing(west,7,8,5,8).
facing(west,7,8,4,8).
facing(west,6,8,5,8).
facing(west,6,8,4,8).
facing(west,5,8,4,8).
facing(west,10,7,10,7).
facing(west,10,7,9,7).
facing(west,10,7,8,7).
facing(west,10,7,7,7).
facing(west,10,7,6,7).
facing(west,10,7,5,7).
facing(west,10,7,4,7).
facing(west,9,7,8,7).
facing(west,9,7,7,7).
facing(west,9,7,6,7).
facing(west,9,7,5,7).
facing(west,9,7,4,7).
facing(west,8,7,7,7).
facing(west,8,7,6,7).
facing(west,8,7,5,7).
facing(west,8,7,4,7).
facing(west,7,7,6,7).
facing(west,7,7,5,7).
facing(west,7,7,4,7).
facing(west,6,7,5,7).
facing(west,6,7,4,7).
facing(west,5,7,4,7).
facing(north_east,4,13,5,14).
facing(north_east,5,13,6,14).
facing(north_east,6,13,7,14).
facing(north_east,7,13,8,14).
facing(north_east,8,13,9,14).
facing(north_east,9,13,10,14).
facing(north_east,4,12,5,13).
facing(north_east,4,12,6,14).
facing(north_east,5,12,6,13).
facing(north_east,5,12,7,14).
facing(north_east,6,12,7,13).
facing(north_east,6,12,8,14).
facing(north_east,7,12,8,13).
facing(north_east,7,12,9,14).
facing(north_east,8,12,9,13).
facing(north_east,8,12,10,14).
facing(north_east,9,12,10,13).
facing(north_east,4,11,5,12).
facing(north_east,4,11,6,13).
facing(north_east,4,11,7,14).
facing(north_east,5,11,6,12).
facing(north_east,5,11,7,13).
facing(north_east,5,11,8,14).
facing(north_east,6,11,7,12).
facing(north_east,6,11,8,13).
facing(north_east,6,11,9,14).
facing(north_east,7,11,8,12).
facing(north_east,7,11,9,13).
facing(north_east,7,11,10,14).
facing(north_east,8,11,9,12).
facing(north_east,8,11,10,13).
facing(north_east,9,11,10,12).
facing(north_east,4,10,5,11).
facing(north_east,4,10,6,12).
facing(north_east,4,10,7,13).
facing(north_east,4,10,8,14).
facing(north_east,5,10,6,11).
facing(north_east,5,10,7,12).
facing(north_east,5,10,8,13).
facing(north_east,5,10,9,14).
facing(north_east,6,10,7,11).
facing(north_east,6,10,8,12).
facing(north_east,6,10,9,13).
facing(north_east,6,10,10,14).
facing(north_east,7,10,8,11).
facing(north_east,7,10,9,12).
facing(north_east,7,10,10,13).
facing(north_east,8,10,9,11).
facing(north_east,8,10,10,12).
facing(north_east,9,10,10,11).
facing(north_east,4,9,5,10).
facing(north_east,4,9,6,11).
facing(north_east,4,9,7,12).
facing(north_east,4,9,8,13).
facing(north_east,4,9,9,14).
facing(north_east,5,9,6,10).
facing(north_east,5,9,7,11).
facing(north_east,5,9,8,12).
facing(north_east,5,9,9,13).
facing(north_east,5,9,10,14).
facing(north_east,6,9,7,10).
facing(north_east,6,9,8,11).
facing(north_east,6,9,9,12).
facing(north_east,6,9,10,13).
facing(north_east,7,9,8,10).
facing(north_east,7,9,9,11).
facing(north_east,7,9,10,12).
facing(north_east,8,9,9,10).
facing(north_east,8,9,10,11).
facing(north_east,9,9,10,10).
facing(north_east,4,8,5,9).
facing(north_east,4,8,6,10).
facing(north_east,4,8,7,11).
facing(north_east,4,8,8,12).
facing(north_east,4,8,9,13).
facing(north_east,4,8,10,14).
facing(north_east,5,8,6,9).
facing(north_east,5,8,7,10).
facing(north_east,5,8,8,11).
facing(north_east,5,8,9,12).
facing(north_east,5,8,10,13).
facing(north_east,6,8,7,9).
facing(north_east,6,8,8,10).
facing(north_east,6,8,9,11).
facing(north_east,6,8,10,12).
facing(north_east,7,8,8,9).
facing(north_east,7,8,9,10).
facing(north_east,7,8,10,11).
facing(north_east,8,8,9,9).
facing(north_east,8,8,10,10).
facing(north_east,9,8,10,9).
facing(north_east,4,7,5,8).
facing(north_east,4,7,6,9).
facing(north_east,4,7,7,10).
facing(north_east,4,7,8,11).
facing(north_east,4,7,9,12).
facing(north_east,4,7,10,13).
facing(north_east,5,7,6,8).
facing(north_east,5,7,7,9).
facing(north_east,5,7,8,10).
facing(north_east,5,7,9,11).
facing(north_east,5,7,10,12).
facing(north_east,6,7,7,8).
facing(north_east,6,7,8,9).
facing(north_east,6,7,9,10).
facing(north_east,6,7,10,11).
facing(north_east,7,7,8,8).
facing(north_east,7,7,9,9).
facing(north_east,7,7,10,10).
facing(north_east,8,7,9,8).
facing(north_east,8,7,10,9).
facing(north_east,9,7,10,8).
facing(north_west,10,13,9,14).
facing(north_west,9,13,8,14).
facing(north_west,8,13,7,14).
facing(north_west,7,13,6,14).
facing(north_west,6,13,5,14).
facing(north_west,5,13,4,14).
facing(north_west,10,12,9,13).
facing(north_west,10,12,8,14).
facing(north_west,9,12,8,13).
facing(north_west,9,12,7,14).
facing(north_west,8,12,7,13).
facing(north_west,8,12,6,14).
facing(north_west,7,12,6,13).
facing(north_west,7,12,5,14).
facing(north_west,6,12,5,13).
facing(north_west,6,12,4,14).
facing(north_west,5,12,4,13).
facing(north_west,10,11,9,12).
facing(north_west,10,11,8,13).
facing(north_west,10,11,7,14).
facing(north_west,9,11,8,12).
facing(north_west,9,11,7,13).
facing(north_west,9,11,6,14).
facing(north_west,8,11,7,12).
facing(north_west,8,11,6,13).
facing(north_west,8,11,5,14).
facing(north_west,7,11,6,12).
facing(north_west,7,11,5,13).
facing(north_west,7,11,4,14).
facing(north_west,6,11,5,12).
facing(north_west,6,11,4,13).
facing(north_west,5,11,4,12).
facing(north_west,10,10,9,11).
facing(north_west,10,10,8,12).
facing(north_west,10,10,7,13).
facing(north_west,10,10,6,14).
facing(north_west,9,10,8,11).
facing(north_west,9,10,7,12).
facing(north_west,9,10,6,13).
facing(north_west,9,10,5,14).
facing(north_west,8,10,7,11).
facing(north_west,8,10,6,12).
facing(north_west,8,10,5,13).
facing(north_west,8,10,4,14).
facing(north_west,7,10,6,11).
facing(north_west,7,10,5,12).
facing(north_west,7,10,4,13).
facing(north_west,6,10,5,11).
facing(north_west,6,10,4,12).
facing(north_west,5,10,4,11).
facing(north_west,10,9,9,10).
facing(north_west,10,9,8,11).
facing(north_west,10,9,7,12).
facing(north_west,10,9,6,13).
facing(north_west,10,9,5,14).
facing(north_west,9,9,8,10).
facing(north_west,9,9,7,11).
facing(north_west,9,9,6,12).
facing(north_west,9,9,5,13).
facing(north_west,9,9,4,14).
facing(north_west,8,9,7,10).
facing(north_west,8,9,6,11).
facing(north_west,8,9,5,12).
facing(north_west,8,9,4,13).
facing(north_west,7,9,6,10).
facing(north_west,7,9,5,11).
facing(north_west,7,9,4,12).
facing(north_west,6,9,5,10).
facing(north_west,6,9,4,11).
facing(north_west,5,9,4,10).
facing(north_west,10,8,9,9).
facing(north_west,10,8,8,10).
facing(north_west,10,8,7,11).
facing(north_west,10,8,6,12).
facing(north_west,10,8,5,13).
facing(north_west,10,8,4,14).
facing(north_west,9,8,8,9).
facing(north_west,9,8,7,10).
facing(north_west,9,8,6,11).
facing(north_west,9,8,5,12).
facing(north_west,9,8,4,13).
facing(north_west,8,8,7,9).
facing(north_west,8,8,6,10).
facing(north_west,8,8,5,11).
facing(north_west,8,8,4,12).
facing(north_west,7,8,6,9).
facing(north_west,7,8,5,10).
facing(north_west,7,8,4,11).
facing(north_west,6,8,5,9).
facing(north_west,6,8,4,10).
facing(north_west,5,8,4,9).
facing(north_west,10,7,9,8).
facing(north_west,10,7,8,9).
facing(north_west,10,7,7,10).
facing(north_west,10,7,6,11).
facing(north_west,10,7,5,12).
facing(north_west,10,7,4,13).
facing(north_west,9,7,8,8).
facing(north_west,9,7,7,9).
facing(north_west,9,7,6,10).
facing(north_west,9,7,5,11).
facing(north_west,9,7,4,12).
facing(north_west,8,7,7,8).
facing(north_west,8,7,6,9).
facing(north_west,8,7,5,10).
facing(north_west,8,7,4,11).
facing(north_west,7,7,6,8).
facing(north_west,7,7,5,9).
facing(north_west,7,7,4,10).
facing(north_west,6,7,5,8).
facing(north_west,6,7,4,9).
facing(north_west,5,7,4,8).
facing(south_east,9,14,10,13).
facing(south_east,8,14,9,13).
facing(south_east,8,14,10,12).
facing(south_east,7,14,8,13).
facing(south_east,7,14,9,12).
facing(south_east,7,14,10,11).
facing(south_east,6,14,7,13).
facing(south_east,6,14,8,12).
facing(south_east,6,14,9,11).
facing(south_east,6,14,10,10).
facing(south_east,5,14,6,13).
facing(south_east,5,14,7,12).
facing(south_east,5,14,8,11).
facing(south_east,5,14,9,10).
facing(south_east,5,14,10,9).
facing(south_east,4,14,5,13).
facing(south_east,4,14,6,12).
facing(south_east,4,14,7,11).
facing(south_east,4,14,8,10).
facing(south_east,4,14,9,9).
facing(south_east,4,14,10,8).
facing(south_east,9,13,10,12).
facing(south_east,8,13,9,12).
facing(south_east,8,13,10,11).
facing(south_east,7,13,8,12).
facing(south_east,7,13,9,11).
facing(south_east,7,13,10,10).
facing(south_east,6,13,7,12).
facing(south_east,6,13,8,11).
facing(south_east,6,13,9,10).
facing(south_east,6,13,10,9).
facing(south_east,5,13,6,12).
facing(south_east,5,13,7,11).
facing(south_east,5,13,8,10).
facing(south_east,5,13,9,9).
facing(south_east,5,13,10,8).
facing(south_east,4,13,5,12).
facing(south_east,4,13,6,11).
facing(south_east,4,13,7,10).
facing(south_east,4,13,8,9).
facing(south_east,4,13,9,8).
facing(south_east,4,13,10,7).
facing(south_east,9,12,10,11).
facing(south_east,8,12,9,11).
facing(south_east,8,12,10,10).
facing(south_east,7,12,8,11).
facing(south_east,7,12,9,10).
facing(south_east,7,12,10,9).
facing(south_east,6,12,7,11).
facing(south_east,6,12,8,10).
facing(south_east,6,12,9,9).
facing(south_east,6,12,10,8).
facing(south_east,5,12,6,11).
facing(south_east,5,12,7,10).
facing(south_east,5,12,8,9).
facing(south_east,5,12,9,8).
facing(south_east,5,12,10,7).
facing(south_east,4,12,5,11).
facing(south_east,4,12,6,10).
facing(south_east,4,12,7,9).
facing(south_east,4,12,8,8).
facing(south_east,4,12,9,7).
facing(south_east,9,11,10,10).
facing(south_east,8,11,9,10).
facing(south_east,8,11,10,9).
facing(south_east,7,11,8,10).
facing(south_east,7,11,9,9).
facing(south_east,7,11,10,8).
facing(south_east,6,11,7,10).
facing(south_east,6,11,8,9).
facing(south_east,6,11,9,8).
facing(south_east,6,11,10,7).
facing(south_east,5,11,6,10).
facing(south_east,5,11,7,9).
facing(south_east,5,11,8,8).
facing(south_east,5,11,9,7).
facing(south_east,4,11,5,10).
facing(south_east,4,11,6,9).
facing(south_east,4,11,7,8).
facing(south_east,4,11,8,7).
facing(south_east,9,10,10,9).
facing(south_east,8,10,9,9).
facing(south_east,8,10,10,8).
facing(south_east,7,10,8,9).
facing(south_east,7,10,9,8).
facing(south_east,7,10,10,7).
facing(south_east,6,10,7,9).
facing(south_east,6,10,8,8).
facing(south_east,6,10,9,7).
facing(south_east,5,10,6,9).
facing(south_east,5,10,7,8).
facing(south_east,5,10,8,7).
facing(south_east,4,10,5,9).
facing(south_east,4,10,6,8).
facing(south_east,4,10,7,7).
facing(south_east,9,9,10,8).
facing(south_east,8,9,9,8).
facing(south_east,8,9,10,7).
facing(south_east,7,9,8,8).
facing(south_east,7,9,9,7).
facing(south_east,6,9,7,8).
facing(south_east,6,9,8,7).
facing(south_east,5,9,6,8).
facing(south_east,5,9,7,7).
facing(south_east,4,9,5,8).
facing(south_east,4,9,6,7).
facing(south_east,9,8,10,7).
facing(south_east,8,8,9,7).
facing(south_east,7,8,8,7).
facing(south_east,6,8,7,7).
facing(south_east,5,8,6,7).
facing(south_east,4,8,5,7).
facing(south_west,10,14,9,13).
facing(south_west,10,14,8,12).
facing(south_west,10,14,7,11).
facing(south_west,10,14,6,10).
facing(south_west,10,14,5,9).
facing(south_west,10,14,4,8).
facing(south_west,9,14,8,13).
facing(south_west,9,14,7,12).
facing(south_west,9,14,6,11).
facing(south_west,9,14,5,10).
facing(south_west,9,14,4,9).
facing(south_west,8,14,7,13).
facing(south_west,8,14,6,12).
facing(south_west,8,14,5,11).
facing(south_west,8,14,4,10).
facing(south_west,7,14,6,13).
facing(south_west,7,14,5,12).
facing(south_west,7,14,4,11).
facing(south_west,6,14,5,13).
facing(south_west,6,14,4,12).
facing(south_west,5,14,4,13).
facing(south_west,10,13,9,12).
facing(south_west,10,13,8,11).
facing(south_west,10,13,7,10).
facing(south_west,10,13,6,9).
facing(south_west,10,13,5,8).
facing(south_west,10,13,4,7).
facing(south_west,9,13,8,12).
facing(south_west,9,13,7,11).
facing(south_west,9,13,6,10).
facing(south_west,9,13,5,9).
facing(south_west,9,13,4,8).
facing(south_west,8,13,7,12).
facing(south_west,8,13,6,11).
facing(south_west,8,13,5,10).
facing(south_west,8,13,4,9).
facing(south_west,7,13,6,12).
facing(south_west,7,13,5,11).
facing(south_west,7,13,4,10).
facing(south_west,6,13,5,12).
facing(south_west,6,13,4,11).
facing(south_west,5,13,4,12).
facing(south_west,10,12,9,11).
facing(south_west,10,12,8,10).
facing(south_west,10,12,7,9).
facing(south_west,10,12,6,8).
facing(south_west,10,12,5,7).
facing(south_west,9,12,8,11).
facing(south_west,9,12,7,10).
facing(south_west,9,12,6,9).
facing(south_west,9,12,5,8).
facing(south_west,9,12,4,7).
facing(south_west,8,12,7,11).
facing(south_west,8,12,6,10).
facing(south_west,8,12,5,9).
facing(south_west,8,12,4,8).
facing(south_west,7,12,6,11).
facing(south_west,7,12,5,10).
facing(south_west,7,12,4,9).
facing(south_west,6,12,5,11).
facing(south_west,6,12,4,10).
facing(south_west,5,12,4,11).
facing(south_west,10,11,9,10).
facing(south_west,10,11,8,9).
facing(south_west,10,11,7,8).
facing(south_west,10,11,6,7).
facing(south_west,9,11,8,10).
facing(south_west,9,11,7,9).
facing(south_west,9,11,6,8).
facing(south_west,9,11,5,7).
facing(south_west,8,11,7,10).
facing(south_west,8,11,6,9).
facing(south_west,8,11,5,8).
facing(south_west,8,11,4,7).
facing(south_west,7,11,6,10).
facing(south_west,7,11,5,9).
facing(south_west,7,11,4,8).
facing(south_west,6,11,5,10).
facing(south_west,6,11,4,9).
facing(south_west,5,11,4,10).
facing(south_west,10,10,9,9).
facing(south_west,10,10,8,8).
facing(south_west,10,10,7,7).
facing(south_west,9,10,8,9).
facing(south_west,9,10,7,8).
facing(south_west,9,10,6,7).
facing(south_west,8,10,7,9).
facing(south_west,8,10,6,8).
facing(south_west,8,10,5,7).
facing(south_west,7,10,6,9).
facing(south_west,7,10,5,8).
facing(south_west,7,10,4,7).
facing(south_west,6,10,5,9).
facing(south_west,6,10,4,8).
facing(south_west,5,10,4,9).
facing(south_west,10,9,9,8).
facing(south_west,10,9,8,7).
facing(south_west,9,9,8,8).
facing(south_west,9,9,7,7).
facing(south_west,8,9,7,8).
facing(south_west,8,9,6,7).
facing(south_west,7,9,6,8).
facing(south_west,7,9,5,7).
facing(south_west,6,9,5,8).
facing(south_west,6,9,4,7).
facing(south_west,5,9,4,8).
facing(south_west,10,8,9,7).
facing(south_west,9,8,8,7).
facing(south_west,8,8,7,7).
facing(south_west,7,8,6,7).
facing(south_west,6,8,5,7).
facing(south_west,5,8,4,7).
goal(I) :- holds(agent_shot(attacker2),I).
holds(in(learner,5,14),0).
holds(face(learner,south),0).
-holds(shot(learner),0).
holds(agent_in(guard2,11,13),0).
holds(agent_face(guard2,south),0).
-holds(shot(guard2),0).
holds(agent_in(guard3,16,15),0).
holds(agent_face(guard3,south),0).
-holds(shot(guard3),0).
holds(agent_in(attacker1,16,6),0).
holds(agent_face(attacker1,north),0).
-holds(shot(attacker1),0).
holds(agent_in(attacker2,10,6),0).
holds(agent_face(attacker2,north),0).
-holds(shot(attacker2),0).
holds(agent_in(attacker3,17,6),0).
holds(agent_face(attacker3,north),0).
-holds(shot(attacker3),0).
holds(in(learner,4,14),1).
holds(face(learner,south),1).
-holds(shot(learner),1).
holds(agent_in(guard2,11,13),1).
holds(agent_face(guard2,south),1).
-holds(shot(guard2),1).
holds(agent_in(guard3,17,15),1).
holds(agent_face(guard3,south),1).
-holds(shot(guard3),1).
holds(agent_in(attacker1,16,7),1).
holds(agent_face(attacker1,north),1).
-holds(shot(attacker1),1).
holds(agent_in(attacker2,10,7),1).
holds(agent_face(attacker2,north),1).
-holds(shot(attacker2),1).
holds(agent_in(attacker3,17,7),1).
holds(agent_face(attacker3,north),1).
-holds(shot(attacker3),1).
display
occurs.
