#!/usr/bin/env python
from __future__ import print_function
# encoding: utf-8
# First Start the server: $> bin/start.py
import argparse
import itertools
import random

import numpy as np
import ad_hoc
import sklweka.jvm as jvm
try:
  import hfo
except ImportError:
  print('Failed to import hfo. To install hfo, in the HFO directory'\
    ' run: \"pip install .\"')
  exit()

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--port', type=int, default=6000, help="Server port")
  parser.add_argument('--seed', type=int, default=None,
                      help="Python randomization seed; uses python default if 0 or not given")
  parser.add_argument('--rand-pass', action="store_true",
                      help="Randomize order of checking teammates for a possible pass")
  parser.add_argument('--epsilon', type=float, default=0,
                      help="Probability of a random action if has the ball, to adjust difficulty")
  parser.add_argument('--record', action='store_true',
                      help="If doing HFO --record")
  parser.add_argument('--rdir', type=str, default='log/',
                      help="Set directory to use if doing --record")
  args=parser.parse_args()
  
  if args.seed:
    random.seed(args.seed)
    
  hfo_env = hfo.HFOEnvironment()
  if args.record:
    # hfo_env.connectToServer(hfo.HIGH_LEVEL_FEATURE_SET,
    #                         'bin/teams/base/config/formations-dt', args.port,
    #                         'localhost', 'base_left', False,
    #                         record_dir=args.rdir)
    # hfo_env.connectToServer(hfo.HIGH_LEVEL_FEATURE_SET,
    #                         'bin/teams/base/config/formations-dt', args.port,
    #                         'localhost', 'HELIOS_left', False,
    #                         record_dir=args.rdir)
    # hfo_env.connectToServer(hfo.HIGH_LEVEL_FEATURE_SET,
    #                         'bin/teams/base/config/formations-dt', args.port,
    #                         'localhost', 'GLIDERS_left', False,
    #                         record_dir=args.rdir)
    # hfo_env.connectToServer(hfo.HIGH_LEVEL_FEATURE_SET,
    #                         'bin/teams/base/config/formations-dt', args.port,
    #                         'localhost', 'CYRUS_left', False,
    #                         record_dir=args.rdir)
    hfo_env.connectToServer(hfo.HIGH_LEVEL_FEATURE_SET,
                            'bin/teams/base/config/formations-dt', args.port,
                            'localhost', 'AXIOM_left', False,
                            record_dir=args.rdir)
    # hfo_env.connectToServer(hfo.HIGH_LEVEL_FEATURE_SET,
    #                         'bin/teams/base/config/formations-dt', args.port,
    #                         'localhost', 'AUT_left', False,
    #                         record_dir=args.rdir)
  else:
    # hfo_env.connectToServer(hfo.HIGH_LEVEL_FEATURE_SET,
    #                         'bin/teams/base/config/formations-dt', args.port,
    #                         'localhost', 'base_left', False)
    # hfo_env.connectToServer(hfo.HIGH_LEVEL_FEATURE_SET,
    #                         'bin/teams/base/config/formations-dt', args.port,
    #                         'localhost', 'HELIOS_left', False)
    # hfo_env.connectToServer(hfo.HIGH_LEVEL_FEATURE_SET,
    #                         'bin/teams/base/config/formations-dt', args.port,
    #                         'localhost', 'GLIDERS_left', False)
    # hfo_env.connectToServer(hfo.HIGH_LEVEL_FEATURE_SET,
    #                         'bin/teams/base/config/formations-dt', args.port,
    #                         'localhost', 'CYRUS_left', False)
    hfo_env.connectToServer(hfo.HIGH_LEVEL_FEATURE_SET,
                           'bin/teams/base/config/formations-dt', args.port,
                           'localhost', 'AXIOM_left', False)
    # hfo_env.connectToServer(hfo.HIGH_LEVEL_FEATURE_SET,
    #                        'bin/teams/base/config/formations-dt', args.port,
    #                        'localhost', 'AUT_left', False)
  num_teammates = hfo_env.getNumTeammates()
  num_opponents = hfo_env.getNumOpponents()
  if args.seed:
    if (args.rand_pass and (num_teammates > 1)) or (args.epsilon > 0):
      print("Python randomization seed: {0:d}".format(args.seed))
    else:
      print("Python randomization seed useless without --rand-pass w/2+ teammates or --epsilon >0")
  if args.rand_pass and (num_teammates > 1):
    print("Randomizing order of checking for a pass")
  if args.epsilon > 0:
    print("Using epsilon {0:n}".format(args.epsilon))
  
  jvm.start() 
  for episode in itertools.count():
    ad_hoc_agent = ad_hoc.Ad_Hoc_Agent(num_teammates, num_opponents)
    num_had_ball = 0
    step = 0
    status = hfo.IN_GAME
    while status == hfo.IN_GAME:
      state = hfo_env.getState()
      action = ad_hoc_agent.get_action(state, step)
      if isinstance(action, tuple):
          hfo_env.act(*action)
      else:
          hfo_env.act(action)
      num_had_ball += 1
      status=hfo_env.step()
      step += 1

    # Quit if the server goes down
    if status == hfo.SERVER_DOWN:
      hfo_env.act(hfo.QUIT)
      exit()
  jvm.stop()
if __name__ == '__main__':
  main()
