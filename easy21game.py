# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 15:46:17 2017

@author: user
"""


import gym
from gym import spaces
from gym.utils import seeding

def cmp(a, b):
    return int((a > b)) - int((a < b))

#negative is red
gameDeck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10]
startDeck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
color = ['red','black']


def draw_game_card(np_random):
    return np_random.choice(gameDeck)
    
def draw_starting_card(np_random):
    return np_random.choice(startDeck)

def draw_color(np_random):
    return np_random.choice(color)


def draw_hand(np_random):
    return [draw_starting_card(np_random)]


def sum_hand(hand):  # Return current hand total
    return sum(hand)


def is_bust(hand):  # Is this hand a bust?
    return sum_hand(hand) > 21 or sum_hand(hand)<1


def score(hand):  # What is the score of this hand (0 if bust)
    return 0 if is_bust(hand) else sum_hand(hand)



class Easy21(gym.Env):

    def __init__(self, natural=False):
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(41),
            spaces.Discrete(10)))
        self._seed()

        # Flag to payout 1.5 on a "natural" blackjack win, like casino rules

        # Start the first game
        self._reset()        # Number of 
        self.nA = 2
        #dealer could reveal a card with 1 to 10
        self.dealer_space=10
        # player has hands from 1 to 21 or bust_hand
        self.player_space=21

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # action = 0 stick
    # action = 1 hit
    def _step(self, action):
        assert self.action_space.contains(action)
        if action:  # hit: add a card to players hand and return
            self.player.append(draw_game_card(self.np_random))
            if is_bust(self.player):
                done = True
                reward = -1
            else:
                done = False
                reward = 0
        else:  # stick: play out the dealers hand, and score
            done = True
            while sum_hand(self.dealer) < 17:
                self.dealer.append(draw_game_card(self.np_random))
            reward = cmp(score(self.player), score(self.dealer))
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        return (sum_hand(self.player), self.dealer[0])

    def _reset(self):
        self.dealer = draw_hand(self.np_random)
        self.player = draw_hand(self.np_random)

        return self._get_obs()
