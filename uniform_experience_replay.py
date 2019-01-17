"""
Created on Wednesday Jan  16 2019

@author: Seyed Mohammad Asghari
@github: https://github.com/s3yyy3d-m
"""

import random
from collections import deque


class Memory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=self.capacity)

    def remember(self, sample):
        self.memory.append(sample)

    def sample(self, n):
        n = min(n, len(self.memory))
        sample_batch = random.sample(self.memory, n)

        return sample_batch