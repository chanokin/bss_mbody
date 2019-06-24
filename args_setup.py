from __future__ import (print_function,
                        unicode_literals,
                        division)
from future.builtins import str, open, range, dict

from spikevo import *
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Mushroom body experiment for classification')

    parser.add_argument('nAL', type=int,
        help='Number of neurons in the Antennae Lobe (first layer)' )
    parser.add_argument('nKC', type=int,
        help='Number of (Kenyon) neurons in the Mushroom body (second layer)' )
    parser.add_argument('nLH', type=int,
        help='Number of inhibitory interneurons in the Lateral Horn (second layer)' )
    parser.add_argument('nDN', type=int,
        help='Number of decision neurons (ouput [third] layer)' )
    parser.add_argument('gScale', type=float,
        help='Global rescaling factor for synaptic strength (weights)' )
    parser.add_argument('w2s', type=float,
        help='Synaptic strength (weights) sufficient to make a neuron spike' )

    parser.add_argument('--backend', '-b', type=str, default=BSS,
        help='PyNN simulator backend')
    
    parser.add_argument('--probAL', type=float, default=0.2,
        help='Probability of active cells in the Antennae Lobe')
    parser.add_argument('--nPatternsAL', type=int, default=10,
        help='Number of patterns for the Antennae Lobe')
    parser.add_argument('--nSamplesAL', type=int, default=1000,
        help='Number of samples from the patterns of the Antennae Lobe')
    parser.add_argument('--randomizeSamplesAL', type=str2bool, default='True',
        help='Randomize the order of samples from the patterns of the Antennae Lobe')
    parser.add_argument('--probNoiseSamplesAL', type=float, default=0.1,
        help='Probability of neurons in the Antennae Lobe flipping state')

    parser.add_argument('--probAL2KC', type=float, default=0.15,
        help='Probability of connectivity between the Antennae Lobe and the Kenyon cells')
    parser.add_argument('--probAL2LH', type=float, default=0.5,
        help='Probability of connectivity between the Antennae Lobe and the Lateral Horn')
    parser.add_argument('--probKC2DN', type=float, default=0.2,
        help='Probability of active weights in the connectivity between '
        'the Kenyon cells and the Decision neurons')
    parser.add_argument('--inactiveScale', type=float, default=0.1,
        help='Scaling for the inactive weights in the connectivity between '
        'the Kenyon cells and the Decision neurons')
    
    parser.add_argument('--renderSpikes', type=str2bool, default='False',
        help='Whether to render spikes from each population. This may slow down getting done with the simulation.')


    parser.add_argument('--regenerateSamples', type=str2bool, default='False',
        help='Whether to generate new samples or grab them from cache files')

    parser.add_argument('--recordAllOutputs', type=str2bool, default='False',
        help='Whether to record as many output variables as possible')

    parser.add_argument('--fixedNumLoops', type=int, default=1,
        help='How many weight recording loops to execute. If zero, 1% of nSamplesAL will be chosen')

    parser.add_argument('--hicann_seed', type=int, default=2,
        help='Random seed to generate placements for populations on BSS chips')

    return parser.parse_args()
