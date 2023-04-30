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

display
occurs.
