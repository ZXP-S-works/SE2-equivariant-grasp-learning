import logging
import threading

SENTINEL = object()


class Pipe:
    def __init__(self, name):
        """
        Consumer-producer pipe.
        Only one variable allowed.
        :param name: the name of the variable
        """
        self.name = name
        self.variable = None
        self.producer_lock = threading.Lock()
        self.consumer_lock = threading.Lock()
        self.consumer_lock.acquire()

    def get_var(self, operator):
        # print(" {}: thread {} about to acquire getlock".format(self.name, operator))
        self.consumer_lock.acquire()
        # print(" {}: thread {} have getlock".format(self.name, operator))
        variable = self.variable
        # print(" {}: thread {} about to release setlock".format(self.name, operator))
        self.producer_lock.release()
        # print(" {}: thread {} setlock released".format(self.name, operator))
        return variable

    def set_var(self, operator, variable):
        # print(" {}: thread {} about to acquire setlock".format(self.name, operator))
        self.producer_lock.acquire()
        # print(" {}: thread {} have setlock".format(self.name, operator))
        self.variable = variable
        # print(" {}: thread {} about to release getlock".format(self.name, operator))
        self.consumer_lock.release()
        # print(" {}: thread {} getlock released".format(self.name, operator))


State = Pipe('state')
Action = Pipe('action')
Reward = Pipe('reward')
Request = Pipe('request')
IsSGDFinished = Pipe('is_SGD_finished')
IsRobotReady = Pipe('is_robot_ready')
# IsSaveModelAnInfo = Pipe('is_save_model_and_info')
