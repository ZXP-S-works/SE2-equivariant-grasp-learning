import numpy as np
import numpy.random as npr
from copy import deepcopy


class QLearningBuffer:
    def __init__(self, size):
        self._storage = []
        self._max_size = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def __getitem__(self, key):
        return self._storage[key]

    def add(self, data):
        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._max_size

    def sample(self, batch_size, onpolicydata=False, onlyfailure=0):
        batch_indexes = npr.choice(self.__len__(), batch_size).tolist()
        if onlyfailure > 0 and self._storage[-1].reward.item() == 0:
            batch_indexes[-onlyfailure:] = np.arange(self.__len__() - onlyfailure, self.__len__())
        batch = [self._storage[idx] for idx in batch_indexes]
        return batch

    def getSaveState(self):
        return {
            'storage': self._storage,
            'max_size': self._max_size,
            'next_idx': self._next_idx
        }

    def loadFromState(self, save_state):
        self._storage = save_state['storage']
        self._max_size = save_state['max_size']
        self._next_idx = save_state['next_idx']


class QLearningBufferExpert(QLearningBuffer):
    def __init__(self, size):
        super().__init__(size)
        self._expert_idx = []

    def add(self, data):
        if self._next_idx >= len(self._storage):
            self._storage.append(data)
            idx = len(self._storage)-1
            self._next_idx = (self._next_idx + 1) % self._max_size
        else:
            self._storage[self._next_idx] = data
            idx = deepcopy(self._next_idx)
            self._next_idx = (self._next_idx + 1) % self._max_size
            while self._storage[self._next_idx].expert:
                self._next_idx = (self._next_idx + 1) % self._max_size
        if data.expert:
            self._expert_idx.append(idx)

    def sample(self, batch_size, onpolicydata=False, onlyfailure=False):
        return super().sample(batch_size, onpolicydata, onlyfailure)
        # if len(self._expert_idx) < batch_size/2 or len(self._storage) - len(self._expert_idx) < batch_size/2:
        #     return super().sample(batch_size, onpolicydata, onlyfailure)
        # expert_indexes = npr.choice(self._expert_idx, int(batch_size / 2)).tolist()
        # non_expert_mask = np.ones(self.__len__(), dtype=np.bool)
        # non_expert_mask[np.array(self._expert_idx)] = 0
        # non_expert_indexes = npr.choice(np.arange(self.__len__())[non_expert_mask], int(batch_size/2)).tolist()
        # if onpolicydata and ((onlyfailure and self._storage[-1].reward.item() == 0) or not onlyfailure):
        #     # change the last index of non_expert_indexes to onpolicy experience
        #     onpolicyindex = npr.randint(1, 9)
        #     non_expert_indexes[-1] = self.__len__() - onpolicyindex
        # batch_indexes = expert_indexes + non_expert_indexes
        # batch = [self._storage[idx] for idx in batch_indexes]
        # return batch

    def getSaveState(self):
        save_state = super().getSaveState()
        save_state['expert_idx'] = self._expert_idx
        return save_state

    def loadFromState(self, save_state):
        super().loadFromState(save_state)
        self._expert_idx = save_state['expert_idx']
