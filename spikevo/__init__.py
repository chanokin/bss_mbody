from __future__ import (print_function,
                        # unicode_literals,
                        division)
from future.builtins import str, open, range, dict

import numpy as np
import sys
import os
import argparse

GENN = 'genn'
NEST = 'nest'
BSS  = 'brainscales'
BSS_BACK = 'pyhmf'
supported_backends = [GENN, NEST, BSS]

RED, GREEN, BLUE = range(3)
ON, OFF = range(2)
CHAN2COLOR = {ON: GREEN, OFF: RED}
CHAN2TXT = {ON: 'GREEN', OFF: 'RED'}
RATE = 'rate'
ON_OFF = 'on-off'
IMAGE_ENCODINGS = [RATE, ON_OFF]
BSS_MAX_SUBPOP_SIZE = 175

def backend_setup(backend):
    if backend.lower() not in supported_backends:
        raise Exception('Backend not supported')

    if backend == GENN:
        import pynn_genn as pynn_local

    elif backend == NEST:
        sys.path.insert(0, os.environ['PYNEST222_PATH'])
        sys.path.insert(0, os.environ['PYNN7_PATH'])
        import pyNN.nest as pynn_local
        
    elif backend == BSS:
        import pyhmf as pynn_local
    
    return pynn_local

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def calc_n_part(size, part_size):
    return size//part_size + int(size % part_size > 0)

def is_string(data):
    return type(data) == type(u'') or isinstance(data, str) or type(data) == type('')

def render_spikes(spikes, title, filename, markersize=1, color='blue'):
    import matplotlib
    try:
        matplotlib.use('Agg')
    except:
        pass
    import matplotlib.pyplot as plt

    n_neurons = len(spikes)
    sys.stdout.write('\n\nRendering spikes: {}\n\n'.format(title))
    sys.stdout.flush()

    fig = plt.figure()
    ax = plt.subplot(1, 1, 1)
    ax.set_title(title)
    total = 0
    for i, times in enumerate(spikes):

        sys.stdout.write('\rNeuron\t{:05d}/{:05d}\t{:06d}'.format(i+1, n_neurons, total))
        sys.stdout.flush()

        plt.plot(times, np.ones_like(times) * i, '.', color=color, markersize=markersize, markeredgewidth=0)
        total += len(times)

    ax.set_ylabel('Neuron id')
    ax.set_xlabel('Time [ms]')

    sys.stdout.write('\nDone!\t{}\n\n'.format(title))
    sys.stdout.flush()

    plt.savefig(filename)
    plt.close(fig)
