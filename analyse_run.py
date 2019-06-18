import numpy as np
import matplotlib.pyplot as plt
import operator
from collections import OrderedDict

def get_high_spiking(spikes, start_t, end_t, min_num_spikes):
    print("\n\nIn get_high_spiking\n\n")
    print(spikes)
    neurons = {}
    
    for nid, times in enumerate(spikes):
        whr = np.where(np.logical_and(start_t <= times, times < end_t))[0]
        
        if len(whr) > min_num_spikes:
            neurons[nid] = len(whr)
    
    print(neurons)
    return sort_by_rate(neurons)

def sort_by_rate(hi_spiking):
    print(hi_spiking.items())
    return OrderedDict(sorted(\
            hi_spiking.items(), key=operator.itemgetter(1), reverse=True))
