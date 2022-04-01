import concurrent.futures
import logging
import threading
import time
import numpy.random as npr
from utils.parallel_utils import *


class Env:
    def __init__(self):
        pass

    def moveHone(self):
        time.sleep(time_scale * 1.5)
        print('robot moved to home')

    def sensor_processing(self):
        while True:
            request = Request.get_var('sensor_processing')
            if request is SENTINEL:
                break
            time.sleep(time_scale * 1)
            obs = npr.random()
            print('obs: ', obs)
            State.set_var('sensor_processing', obs)
        print('sensor_processing killed')

    def picking(self, action):
        print('pick at: ', action)
        time.sleep(time_scale * 3.6)
        print('finished picking')

    def move_reward(self):
        time.sleep(time_scale * 1.5)
        reward = npr.randint(low=0, high=2)
        print('moved to the center, reward: ', reward)
        Reward.set_var('move_reward', reward)

    def place_move_center(self, is_request=True):
        time.sleep(time_scale * 0.2)
        if is_request:
            Request.set_var('place', 1)
        time.sleep(time_scale * 2.4)
        print('robot is ready for picking')


def main():
    env.moveHone()
    Request.set_var('main', 1)
    for steps in range(max_episode):
        action = Action.get_var('main')
        env.picking(action)
        env.move_reward()
        env.place_move_center(is_request=(steps != max_episode - 1))
        IsSGDFinished.get_var('main')
        print('--------finished step', steps + 1, '---------')
    State.set_var('main', SENTINEL)
    Action.set_var('main', SENTINEL)
    Reward.set_var('main', SENTINEL)
    Request.set_var('main', SENTINEL)
    env.moveHone()
    print('training finished')


class Agent:
    def __init__(self):
        self.obs = None
        self.network = threading.Lock()
        pass

    def get_action(self):
        while True:
            state = State.get_var('get_action')
            if state is SENTINEL:
                break
            self.network.acquire()
            self.obs = state
            time.sleep(time_scale * 0.5)
            a = npr.random()
            print('agent s action', a)
            Action.set_var('get_action', a)
            self.network.release()
        print('get_action killed')

    def store_transition_SGD(self):
        while True:
            reward = Reward.get_var('store_transition_SGD')
            if reward is SENTINEL or self.obs is SENTINEL:
                break
            self.network.acquire()
            print('get obs: {}, reward: {}'.format(self.obs, reward))
            time.sleep(time_scale * 2)
            print('transition augmented')
            time.sleep(time_scale * 1)
            print('trained a SGD step')
            IsSGDFinished.set_var('store_transition_SGD', True)
            self.network.release()
        print('store_transition_SGD killed')


if __name__ == '__main__':
    format_ = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format_, level=logging.INFO,
                        datefmt="%H:%M:%S")
    logging.getLogger().setLevel(logging.DEBUG)

    max_episode = 10
    time_scale = 0.1
    env = Env()
    agent = Agent()
    State = Pipe('state')
    Action = Pipe('action')
    Reward = Pipe('reward')
    Request = Pipe('request')
    IsSGDFinished = Pipe('is_SGD_finished')

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        executor.submit(main)
        executor.submit(env.sensor_processing)
        executor.submit(agent.get_action)
        executor.submit(agent.store_transition_SGD)

