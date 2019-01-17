"""
Created on Wednesday Jan  16 2019

@author: Seyed Mohammad Asghari
@github: https://github.com/s3yyy3d-m
"""

import random
from sum_tree import SumTree as ST


class Memory(object):
    e = 0.05

    def __init__(self, capacity, pr_scale):
        self.capacity = capacity
        self.memory = ST(self.capacity)
        self.pr_scale = pr_scale
        self.max_pr = 0

    def get_priority(self, error):
        return (error + self.e) ** self.pr_scale

    def remember(self, sample, error):
        p = self.get_priority(error)

        self_max = max(self.max_pr, p)
        self.memory.add(self_max, sample)

    def sample(self, n):
        sample_batch = []
        sample_batch_indices = []
        sample_batch_priorities = []
        num_segments = self.memory.total() / n

        for i in xrange(n):
            left = num_segments * i
            right = num_segments * (i + 1)

            s = random.uniform(left, right)
            idx, pr, data = self.memory.get(s)
            sample_batch.append((idx, data))
            sample_batch_indices.append(idx)
            sample_batch_priorities.append(pr)

        return [sample_batch, sample_batch_indices, sample_batch_priorities]

    def update(self, batch_indices, errors):
        for i in xrange(len(batch_indices)):
            p = self.get_priority(errors[i])
            self.memory.update(batch_indices[i], p)