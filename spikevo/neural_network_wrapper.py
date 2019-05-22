import future
import builtins
import past
import six


from __future__ import print_function, 
                        # unicode_literals, 
                        absolute_import
from builtins import range, map, object

import numpy as np


class NeuralNetworkWrapper(object):
    def __init__(self, architecture):
        pass
    def get_weights_shapes(self):
        pass
    def set_weights(self, weight_matrices):
        pass
    def score(self, x, y):
        """do the actual run, probably a good place to call netexec?"""
        pass
