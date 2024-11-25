---
layout: post
title:  "Grid World Env 구현"
---

# Discrete Env 만들기


```python
import io
import numpy as np
import sys
from copy import deepcopy as dc

import numpy as np

from gym import Env, spaces
from gym.utils import seeding

def categorical_sample(prob_n, np_random):
    """
    Sample from categorical distribution
    Each row specifies class probabilities
    """
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np_random.rand()).argmax()

class DiscreteEnv(Env):
    """
    Has the following members
    - nS: number of states
    - nA: number of actions
    - P: transitions (*)
    - isd: initial state distribution (**)
    (*) dictionary dict of dicts of lists, where
      P[s][a] == [(probability, nextstate, reward, done), ...]
    (**) list or array of length nS
    """
    def __init__(self, nS, nA, P, isd):
        self.P = P
        self.isd = isd
        self.lastaction=None # for rendering
        self.nS = nS
        self.nA = nA
        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Discrete(self.nS)

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.s = categorical_sample(self.isd, self.np_random)
        self.lastaction=None
        return self.s

    def step(self, a):
        transitions = self.P[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, d= transitions[i]
        self.s = s
        self.lastaction=a
        return (s, r, d, {"prob" : p})
```

# Gridworld Env


```python
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

metadata = {'render.modes': ['human', 'ansi']}

class GridworldEnv(DiscreteEnv):
    def __init__(self, shape=[4, 4]):
        self.shape = shape
        nS = np.prod(shape)     # #state:16
        nA = 4                  # #action
        MAX_Y = shape[0]        # 4
        MAX_X = shape[1]        # 4

        P = {}                  # state,action별(p,s',r,T?)저장 공간
        grid = np.arange(nS).reshape(shape) # (4,4)
        it = np.nditer(grid, flags=['multi_index']) # state->iterator
        while not it.finished:  ## state값 초기화
            s = it.iterindex        # state idx(x)
            y, x = it.multi_index   # state idx(y,x)

            ## P[s][a]=P[s]{a:[(prob, next_state, reward, is_done)]}=P{s:{a:[()]}}
            P[s] = {a: [] for a in range(nA)}   # state/action list 초기화

            def is_done(s): return s == 0 or s == (nS - 1)  #Terminal state
            reward = 0.0 if is_done(s) else -1.0            #T면 0,아니면 -1

            if is_done(s): # terminal state면 상태 유지
                P[s][UP] = [(1.0, s, reward, True)]
                P[s][RIGHT] = [(1.0, s, reward, True)]
                P[s][DOWN] = [(1.0, s, reward, True)]
                P[s][LEFT] = [(1.0, s, reward, True)]
            else:          # action별 상태값[(p,s',r,T?)] setting
                ns_up = s if y == 0 else s - MAX_X               # action별로
                ns_right = s if x == (MAX_X - 1) else s + 1      # s'
                ns_down = s if y == (MAX_Y - 1) else s + MAX_X   # 결정 후
                ns_left = s if x == 0 else s - 1                 # action별
                P[s][UP] = [(1.0, ns_up, reward, is_done(ns_up))]# state값 set
                P[s][RIGHT] = [(1.0, ns_right, reward, is_done(ns_right))]
                P[s][DOWN] = [(1.0, ns_down, reward, is_done(ns_down))]
                P[s][LEFT] = [(1.0, ns_left, reward, is_done(ns_left))]
            it.iternext()
        self.P = P   # P{s:{a:[(p,s',r,T?)]}}
        self.P_tensor = np.zeros(shape=(nA, nS, nS)) # 상태천이행렬(4,16,16)<-0
        self.R_tensor = np.zeros(shape=(nS, nA))     # 보상(16,16)<-0
        ## P_t/R_t <- P
        for s in self.P.keys():                          # P[s] loop
            for a in self.P[s].keys():                   # P[s][a] loop
                p_sa, s_prime, r, done = self.P[s][a][0] # <-(p,s',r,T?)
                self.P_tensor[a, s, s_prime] = p_sa      # P_t[a,s,s']<-p
                self.R_tensor[s, a] = r                  # R_t[s,a]<- r
        isd = np.ones(nS) / nS # 상태별 초기 확률을 1/16로 설정
        #상위 class methode사용가능하도록 init 호출
        super(GridworldEnv, self).__init__(nS, nA, P, isd)
        #

    def observe(self):
        return dc(self.s) #deepcopy

    def render(self, mode='human', close=False):
        """ ==========
            T  o  o  o
            o  x  o  o
            o  o  o  o
            o  o  o  T
            =========="""
        if close: return

        outfile = io.StringIO() if mode == 'ansi' else sys.stdout
        grid = np.arange(self.nS).reshape(self.shape)
        it = np.nditer(grid, flags=['multi_index'])

        outfile.write('==' * self.shape[1] + '==\n')

        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index

            if self.s == s: # hole
                output = " x "
            elif s == 0 or s == self.nS - 1:
                output = " T "
            else:
                output = " o "

            if x == 0:
                output = output.lstrip() # removes any leading characters
            if x == self.shape[1] - 1:
                output = output.rstrip()

            outfile.write(output)

            if x == self.shape[1] - 1:
                outfile.write("\n")

            it.iternext()

        outfile.write('==' * self.shape[1] + '==\n')
```
