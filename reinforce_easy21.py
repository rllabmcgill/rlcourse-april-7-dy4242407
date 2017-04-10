#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 15:26:56 2017

@author: yuedong
"""
import os
os.chdir("/Users/yuedong/Downloads/comp767_assignment_5/")
#%%

from easy21game import Easy21
import numpy as np
from matplotlib import cm
#%%

def dealerFeatures(dealerState):
    dealerVec = np.array([dealerState in [1,2,3,4], dealerState in [4,5,6,7], 
                   dealerState in [7,8,9,10]]).astype('int')
    return np.where(dealerVec)[0]
  
    
# INPUT
#   playerState: sum of the player, integer between 1 and 21
# OUTPUT
#   boolean vector coding the player card interval on 6 bits
def playerFeatures(playerState):
    playerVec = np.array([playerState in [1,2,3,4,5,6], 
                     playerState in [4,5,6,7,8,9],
                     playerState in [7,8,9,10,11,12],
                     playerState in [10,11,12,13,14,15],
                     playerState in [13,14,15,16,17,18],
                     playerState in [16,17,18,19,20,21]]).astype('int')
    return np.where(playerVec)[0]


# INPUTS
#   s: state =(playerState,dealerState) (as defined in env._step)
#   a: action, integer: HIT(1) or STICK(0)
# returns a binary vector of length 36 representing the features
def phi(s, a):
    tmp = np.zeros(shape=(3,6,2)) #zeros array of dim 3*6*2
    #putting one where a feature is on
    for i in dealerFeatures(s[1]):
        for j in playerFeatures(s[0]):
            tmp[i,j,a] = 1 
    return(tmp.flatten()) #returning 'vectorized' (1-dim) array
#%%
# here is the code for the policy approximation
    
# this function calculates h(s,a, theta)
def preference_cal(theta, state, action):
    h_s_a = np.dot(theta, phi(state,action))
    return h_s_a
    
# this function computes pi(a|s,theta) given the action_set
def policy_prob(theta, state, action, action_set):
        
    numerator = np.exp(preference_cal(theta, state, action))
        
    denominators = []
    for a in action_set:
        denominators.append(np.exp(preference_cal(theta, state, a)))
    denominator_sum = sum(denominators)
        
    pi = numerator / denominator_sum
    return  pi
    
# this function returns an action, where the action a is chosen with prob pi(a|s,theta)
# using policy parametrization
def policy_par(theta, state, action_set):
    
    pi_actions = []
    for a in action_set:
        pi_actions.append(policy_prob(theta, state, a, action_set))
            
    return np.random.choice(action_set, p= pi_actions)

#%%
# actor-critic with eligibility trace

class REINFORCE_easy21:
    def __init__(self, environment, gamma=1, alpha=0.01):

        self.env = environment
        self.gamma = gamma
        self.alpha = alpha

        
        self.theta = np.zeros(36) 
        # o stick, 1 hit
        self.action_set = [0,1]
        
        self.iterations = 0
        self.returns = []
    
    def train(self, iterations):        
        # Loop episodes
        for episode in range(iterations):
            
            #initialize trajactory for states, actions, rewards within one episode
            states = []
            actions = []
            rewards = []
            
            # get initial state for current episode
            s = self.env._reset()
            states.append(s)
            
            a = policy_par(self.theta, s, self.action_set)
            actions.append(a)
            
            term = False
            time_step = 0

            
            # generate an episode untill the end
            while not term:
                
                # execute action
                s_next, r, term = self.env._step(a)[0:3]
                rewards.append(r)

                # reassign s and a, add to trajactory
                s = s_next
                states.append(s)
                
                a = policy_par(self.theta, s, self.action_set)
                actions.append(a)
                
                time_step += 1 
            
            # update theta
            for t in range(time_step):
               G_t = np.sum(rewards[t:])
               # if gamma != 1, add np.power(self.gamma, t)
               gradient = phi(states[t],actions[t])
               
               gradient = phi(states[t],actions[t])-sum(
                       [policy_prob(self.theta, states[t], a, self.action_set) * 
                                 phi(states[t], a) for a in self.action_set])
                             
               self.theta += self.alpha * G_t * gradient
            
            self.returns.append(sum(rewards))
            
        self.iterations += iterations
        
#%%
env = Easy21()
agent = REINFORCE_easy21(env)
#%%
for i in range (1):
    agent.train(10000)
