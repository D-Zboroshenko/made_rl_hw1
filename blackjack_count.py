import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

def is_natural(hand):  # Is this hand a natural blackjack?
    return sorted(hand) == [1, 10]

def shuffle_deck(num_decks = 1):
    deck = [num_decks * 4 for i in range(9)]
    deck.append(num_decks * 16)
    return np.array(deck)

def usable_ace(hand):
    return 1 in hand and sum(hand) + 10 <= 21

def sum_hand(hand):
    if usable_ace(hand):
        return sum(hand) + 10
    return sum(hand)

def is_bust(hand):
    return sum_hand(hand) > 21

def score(hand):
    return 0 if is_bust(hand) else sum_hand(hand)

def cmp(a, b):
    return float(a > b) - float(a < b)

class BlackjackEnv(gym.Env):
    def __init__(self, natural=False):
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(32),
            spaces.Discrete(11),
            spaces.Discrete(2),
            spaces.Discrete(21)))
        self.natural = natural
        self.shuffle()
        self.points = np.array([[-1, 1, 1, 1, 1, 1, 0, 0, 0, -1]])
        self.reset()
    
    def shuffle(self):
        self.deck = shuffle_deck()
        
    def draw_card(self):
        card = np.random.choice(np.arange(1, 11), p=self.deck / self.deck.sum())
        self.deck[card - 1] -= 1
        
        return card
    
    def draw_hand(self):
        return [self.draw_card(), self.draw_card()]
        
    def step(self, action):
        assert self.action_space.contains(action)
        if action == 1:
            self.player.append(self.draw_card())
            if is_bust(self.player):
                done = True
                reward = -1.
            else:
                done = False
                reward = 0.
        if action == 2:
            self.player.append(self.draw_card())
            done = True
            if is_bust(self.player):
                reward = -2.
            else:
                while sum_hand(self.dealer) < 17:
                    self.dealer.append(self.draw_card()) 
                reward = cmp(score(self.player), score(self.dealer)) * 2.
        else:
            done = True
            while sum_hand(self.dealer) < 17:
                self.dealer.append(self.draw_card())
            reward = cmp(score(self.player), score(self.dealer))
            if self.natural and is_natural(self.player) and reward == 1.:
                reward = 1.5
            
        if (sum(self.deck) < 15) & (done):
            self.shuffle()
            
        return self._get_obs(), reward, done, {}
    
    def _get_obs(self):
        counter = shuffle_deck() - self.deck
        for i in self.dealer[1:]:
            counter[i - 1] -= 1
        counter = self.points.dot(counter)[0]
        return (sum_hand(self.player), self.dealer[0], usable_ace(self.player), counter)
    
    def reset(self):
        self.dealer = self.draw_hand()
        self.player = self.draw_hand()
        return self._get_obs()