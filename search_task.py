# coding: utf-8
from enum import Enum
import numpy as np
import random
import time


class State():
    def __init__(self, pos=[0, 0]):
        self.__pos = np.array(pos)

    @property
    def pos(self):
        return self.__pos

    @pos.setter
    def pos(self, value):
        self.__pos = value

    @property
    def id(self):
        return self.__pos[0], self.__pos[1]

    def clone(self):
        return State(self.pos)


class Action():
    Up = np.array([1, 0])
    Down = np.array([-1, 0])
    Left = np.array([0, 1])
    Right = np.array([0, -1])

    @classmethod
    def actions(cls):
        return [Action.Up, Action.Down, Action.Left, Action.Right]

    @classmethod
    def inverse(cls, dir):
        return dir * -1


class Attribute(Enum):
    Plain = (-0.04, False)
    Wall = (0, False)
    Goal = (1, True)
    Hole = (-1, True)


class Agent():
    actions = Action.actions()
    action_len = len(actions)

    def policy(self, state):
        '''
        <<戦略>>
        状態を元に行動を決定する
        '''
        return random.choice(Agent.actions)


class Environment():
    def __init__(self, grid, move_prob=0.8):
        self.__grid = grid
        self.__move_prob = move_prob
        self.__other_prob = (1 - move_prob) / (len(Action.actions())-2)
        self.__state = None

    @property
    def max_grid(self):
        return np.array(self.__grid.shape)-1

    def reset(self):
        self.__state = State()
        return self.__state

    def step(self, state, action):
        probs = self.transit_probs(state, action)

        if len(probs) == 0:
            return None, 0, True

        next_state = np.random.choice(
            list(probs.keys()), p=list(probs.values()))
        reward, done = self.reward(next_state)

        self.__state = next_state

        return next_state, reward, done

    def transit_probs(self, state, action):
        '''
        <<遷移関数>>
        現在の状態と行動に対して各ステートに遷移する確率を返す
        '''
        probs = {}

        if not self.can_transit_from(state):
            print("Already on the terminal cell.")
            return probs

        opposit_dir = Action.inverse(action)

        for a in Action.actions():
            prob = self.__move_prob if np.all(a == action) else \
                self.__other_prob if np.all(a != opposit_dir) else 0

            next_state = self.__move(state, a)
            if next_state in probs:
                probs[next_state] += prob
            else:
                probs[next_state] = prob

        return probs

    def reward(self, state):
        '''
        <<報酬関数>>
        Attributeに対応する報酬を返す
        '''
        return self.__grid[state.id].value

    def can_transit_from(self, state):
        return np.all(self.__grid[state.id] == Attribute.Plain)

    def __move(self, state, action):
        '''
        grid上を移動する
        '''
        if not self.can_transit_from(state):
            raise Exception("Can't move from here.")

        next_state = state.clone()
        next_state.pos = np.clip(
            next_state.pos + action, 0, self.max_grid)

        if self.__grid[next_state.id] == Attribute.Wall:
            next_state = state

        # print("action:{}, pos:{}".format(action, next_state.pos))
        return next_state


if __name__ == "__main__":
    a = Attribute
    grid = np.array([
        [a.Plain, a.Plain, a.Plain, a.Goal],
        [a.Plain, a.Wall, a.Plain, a.Hole],
        [a.Plain, a.Plain, a.Plain, a.Plain]
    ])

    env = Environment(grid)
    agent = Agent()

    for i in range(10):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.policy(state)
            next_state, reward, done = env.step(state, action)
            total_reward += reward
            state = next_state

        print("Episode {}:Agent gets {} reward.".format(i, total_reward))
