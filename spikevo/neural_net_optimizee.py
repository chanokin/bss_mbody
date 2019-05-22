import future
import builtins
import past
import six


from __future__ import print_function, 
                        # unicode_literals, 
                        absolute_import
from builtins import range, map, object


from collections import namedtuple

import numpy as np
from sklearn.datasets import load_digits, fetch_mldata

from l2l.optimizees.optimizee import Optimizee
from .neural_network_wrapper import NeuralNetworkWrapper


class NNWOptimizee(Optimizee):
    def __init__(self, trajectory, parameters):
        super().__init__(trajectory)
        ### build NeuralNetworkWrapper object given params
        ### add params to trajectory
        self.nnw = NeuralNetworkWrapper({}) # todo: give actual architecture!
        self.inputs = None # todo: load actual inputs!
        self.labels = None # todo: load actual labels!

    def create_individual(self):
        """generate instance optimizables (e.g. weights for the network)"""
        pass
        #return dict(weights=flattened_weights)

    def bounding_func(self, individual):
        """
        Bounds the individual within the required bounds via coordinate clipping
        """
        ### don't really know what this does! it doesn't seem to be bounding a thing!
        return individual

    def simulate(self, trajectory):
        """evaluate the network given parameters set in the trajectory"""
        return self.nnw.score(self.inputs, self.labels)
