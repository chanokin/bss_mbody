#!/usr/bin/env python
# -*- coding: utf-8; -*-
from __future__ import (print_function,
                        # unicode_literals,
                        division)
from future.builtins import open, range, dict

import numpy as np
import matplotlib.pyplot as plt
import sys
import time
import datetime
from spikevo import *
from spikevo.pynn_transforms import PyNNAL
from spikevo.wafer import Wafer as WAL
import argparse
from pprint import pprint
from args_setup import get_args
from input_utils import *
from analyse_run import get_high_spiking

# from pyhalbe import HICANN
import pyhalbe.Coordinate as C
# from pysthal.command_line_util import init_logger
# from pymarocco import PyMarocco
# from pymarocco.runtime import Runtime
# from pymarocco.coordinates import LogicalNeuron
# from pymarocco.results import Marocco
# from pymarocco import Defects
# from pysthal.command_line_util import init_logger
# import pyhmf as pynn_local

# init_logger("WARN", [
    # ("guidebook", "INFO"),
    # ("marocco", "INFO"),
    # ("Calibtic", "INFO"),
    # ("sthal", "INFO")
# ])


def get_hicanns(center_hicann, n_kenyon):
    np.random.seed(2)
    ID, ROW, COL = range(3)
    w = WAL()
    hood = w.get_neighbours(center_hicann, max_dist=3)
    ids = []
    for r in hood:
        for c in hood[r]:
            ids.append(hood[r][c][ID])

    pprint(hood)
    
    pops = ['antenna', 'decision', 
            'feedback', 'exciter',
            # 'tick', 'exciter_src',
            # 'kenyon',
            ]
    ### ideal config is in a 3x3 grid
    places = {}
    used = []
    for p in pops:
        avail = np.setdiff1d(ids, used)
        hicann_id = np.random.choice(avail, size=3)
        hicann = [C.HICANNOnWafer(C.Enum(i)) for i in hicann_id]
        places[p] = hicann
        for i in hicann_id:
            used.append(i)
            
    k_places = []
    for i in range(n_kenyon):
        avail = np.setdiff1d(ids, used)
        np.random.choice(avail, size=n_kenyon, replace=False)
        hicann_id = np.random.choice(avail, size=3)

        hicann = [C.HICANNOnWafer(C.Enum(i)) for i in hicann_id]
        for i in hicann_id:
            used.append(i)
        k_places.append(hicann)
        
    places['kenyon'] = k_places

    # places = {
    # 'antenna': None,
    # 'kenyon': [None] * n_kenyon,
    # 'decision': None,
    # 'tick': None,
    # 'feedback': None,
    # 'exciter src': None,
    # 'exciter': None,
    # }
    
    for k in sorted(places):
        for p in places[k]:
            try:
                print(k, p.id())
            except:
                for q in p:
                    print(k, q.id())
    return places

def gain_control_list(input_size, horn_size, max_w, cutoff=0.7):
    if cutoff is not None:
        n_cutoff = int(cutoff * horn_size)
    else:
        n_cutoff = 15
    matrix = np.ones((input_size * horn_size, 4))
    matrix[:, 0] = np.repeat(np.arange(input_size), horn_size)
    matrix[:, 1] = np.tile(np.arange(horn_size), input_size)

    matrix[:, 2] = np.tile(max_w / (n_cutoff + 1.0 + np.arange(horn_size)), input_size)

    return matrix


def output_connection_list(kenyon_size, decision_size, prob_active, active_weight,
                           inactive_scaling, delay=1, seed=1, clip_to=np.inf):
    matrix = np.ones((kenyon_size * decision_size, 4)) * delay
    matrix[:, 0] = np.repeat(np.arange(kenyon_size), decision_size)
    matrix[:, 1] = np.tile(np.arange(decision_size), kenyon_size)

    np.random.seed(seed)

    inactive_weight = active_weight * inactive_scaling
    matrix[:, 2] = np.clip(np.random.normal(inactive_weight, inactive_weight * 0.2,
                                            size=(kenyon_size * decision_size)),
                           0, clip_to)

    dice = np.random.uniform(0., 1., size=(kenyon_size * decision_size))
    active = np.where(dice <= prob_active)[0]
    matrix[active, 2] = np.clip(np.random.normal(active_weight, active_weight * 0.2,
                                                 size=active.shape),
                                0, clip_to)
    
    n_above = (matrix[:, 2] > (inactive_weight + active_weight)/2.0).sum()
    n_total = float(matrix[:, 2].size)
    print("n_above, n_total, n_above/n_total")
    print(n_above, n_total, n_above/n_total)
    np.random.seed()
    # pprint(np.where(matrix[:, 2] < 0.0))
    # return matrix
    conn_list = [[int(pre), int(post), w, d] for pre, post, w, d in matrix]
    return conn_list


def output_pairing_connection_list(decision_size, neighbour_distance, weight, delay=1):
    conn_list = []
    half_dist = neighbour_distance // 2
    for nid in range(decision_size):
        for ndist in range(-half_dist, half_dist + 1):
            neighbour = nid + ndist
            if neighbour < 0 or ndist == 0 or neighbour >= decision_size:
                continue
            conn_list.append([nid, neighbour, weight, delay])

    return conn_list

def args_to_str(arguments, stringable=['nAL', 'nKC', 'nDN', 'probAL', 'probNoiseSamplesAL', 'nPatternsAL']):

    d = vars(arguments)
    arglist = []
    for arg in d:
        v = str(d[arg])
        if arg not in stringable:
            continue
        v = v.replace('.', 'p')
        arglist.append('{}_{}'.format(arg, v))

    return '__'.join(arglist)


args = get_args()
pprint(args)

backend = args.backend

neuron_class = 'IF_cond_exp'
# heidelberg's brainscales seems to like these params

e_rev = 92  # mV
# e_rev = 500.0 #mV

base_params = {
    'cm': 0.01,  # nF
    # 'cm': 0.2,  # nF
    'v_reset': -70.,  # mV
    'v_rest': -65.,  # mV
    'v_thresh': -58.,  # mV
    # 'v_thresh': -50.,  # mV
    # 'e_rev_I': -e_rev, #mV
    # 'e_rev_E': 0.,#e_rev, #mV
    # 'tau_m': 10.,  # ms
    'tau_m': 10.,  # ms
    'tau_refrac': 10.,  # ms
    'tau_syn_E': 10.0,  # ms
    'tau_syn_I': 10.0,  # ms

}

base_params['e_rev_I'] = -e_rev
# base_params['e_rev_E'] = e_rev
base_params['e_rev_E'] = 0.0


kenyon_parameters = base_params.copy()
# kenyon_parameters['tau_refrac'] = 1.0  # ms
# kenyon_parameters['tau_syn_E'] = 1.0  # ms
# kenyon_parameters['tau_syn_I'] = 1.0  # ms
# kenyon_parameters['tau_m'] = 10.0  # ms


horn_parameters = base_params.copy()
# horn_parameters['tau_m'] = 5.0
# horn_parameters['tau_syn_E'] = 1.0  # ms

decision_parameters = base_params.copy()
# decision_parameters['tau_syn_E'] = 1.0  # ms
# decision_parameters['tau_syn_I'] = 2.0  # ms
# decision_parameters['tau_refrac'] = 15.0
# decision_parameters['tau_m'] = 5.0
# decision_parameters['v_reset'] = -100.0

fb_parameters = base_params.copy()
# fb_parameters['tau_syn_E'] = 1.0  # ms
# fb_parameters['tau_syn_I'] = 1.0  # ms
# fb_parameters['tau_refrac'] = 10.0


exciter_parameters = base_params.copy()
# exciter_parameters['tau_refrac'] = 1.0  # ms
# exciter_parameters['tau_syn_E'] = 1.0  # ms
# exciter_parameters['tau_syn_I'] = 100.0  # ms
# exciter_parameters['tau_m'] = 5.0  # ms
# exciter_parameters['v_reset'] = -65.0  # ms

neuron_params = {
    'base': base_params, 'kenyon': kenyon_parameters,
    'horn': horn_parameters, 'decision': decision_parameters,
    'feedback': fb_parameters, 'exciter': exciter_parameters,
}

# W2S = args.w2s
# W2S = 0.0000001
W2S = 1.0

# sample_dt, start_dt, max_rand_dt = 10, 5, 2
sample_dt, start_dt, max_rand_dt = 50, 5, 5.0
sim_time = sample_dt * args.nSamplesAL * args.nPatternsAL
timestep = 0.1
regenerate = args.regenerateSamples
record_all = args.recordAllOutputs and args.nSamplesAL <= 50
fixed_loops = args.fixedNumLoops
n_explore_samples = min(args.nPatternsAL * 10, np.round(args.nSamplesAL * args.nPatternsAL * 0.01))
n_exciter_samples = min(args.nPatternsAL * 100, np.round(args.nSamplesAL * args.nPatternsAL * 0.1))
n_test_samples = min(1000, np.round(args.nSamplesAL * args.nPatternsAL * 1.0/6.0))
use_poisson_input = bool(0)
high_dt = 3
low_freq, high_freq = 10, 100

sys.stdout.write('Creating input patterns\n')
sys.stdout.flush()

sys.stdout.write('\tGenerating input vectors\n')
sys.stdout.flush()

input_vecs = generate_input_vectors(args.nPatternsAL, args.nAL, args.probAL, seed=123)
# input_vecs = generate_input_vectors(10, 100, 0.1)
# pprint(input_vecs)
sys.stdout.write('\t\tDone with input vectors\n')
sys.stdout.flush()


sys.stdout.write('\tGenerating samples\n')
sys.stdout.flush()

samples = generate_samples(input_vecs, args.nSamplesAL, args.probNoiseSamplesAL, seed=234,
                           # method='random',
                           method='exact',
                           regenerate=regenerate)
# pprint(samples)
sys.stdout.write('\t\tdone with samples\n')
sys.stdout.flush()


sys.stdout.write('\tGenerating spike times\n')
sys.stdout.flush()

if use_poisson_input:
    sample_indices, spike_times = generate_spike_times_poisson(input_vecs, samples,
                                    sample_dt, start_dt, high_dt, high_freq, low_freq,
                                    seed=234, randomize_samples=True, regenerate=bool(0))
else:
    sample_indices, spike_times = samples_to_spike_times(samples, sample_dt, start_dt, max_rand_dt, timestep,
                                                         randomize_samples=args.randomizeSamplesAL, seed=345,
                                                         regenerate=regenerate)




tick_spikes = generate_tick_spikes(samples, sample_dt, start_dt, n_test_samples, delay=25)

sys.stdout.write('\t\tdone with spike times\n')
sys.stdout.flush()

sys.stdout.write('Done!\tCreating input patterns\n\n')
sys.stdout.flush()

if args.renderSpikes:
    render_spikes(spike_times, 'Input samples', 'input_samples.pdf', markersize=1)
# plt.show()

### -------------------------------------------------------------- ###
### -------------------------------------------------------------- ###
### -------------------------------------------------------------- ###

sys.stdout.write('Creating simulator abstraction\n')
sys.stdout.flush()



pynnx = PyNNAL(backend)
pynnx.setup(timestep=timestep, min_delay=timestep, 
            per_sim_params={'wafer': 30})

sys.stdout.write('Done!\tCreating simulator abstraction\n\n')
sys.stdout.flush()

sys.stdout.write('Creating populations\n')
sys.stdout.flush()

#######################################################################
#######################################################################
#######################################################################

div_kc = 5
central_hicann = 76
# central_hicann = 107
# central_hicann = 171
# central_hicann = 283
# central_hicann = 275
hicanns = get_hicanns(central_hicann, div_kc)
# pprint(hicanns)
nkc = int(np.ceil(args.nKC/float(div_kc)))
print("\n\nnumber of neurons in per kenyon subpop = {}\n".format(nkc))

#######################################################################
#######################################################################
#######################################################################
#######################################################################

populations = {
    'antenna': pynnx.Pop(args.nAL, 'SpikeSourceArray',
                         {'spike_times': spike_times}, label='Antennae Lobe',
                         hicann=hicanns['antenna']),
    'decision': pynnx.Pop(args.nDN, neuron_class,
                          decision_parameters, label='Decision Neurons',
                          hicann=hicanns['decision'],
                          gmax=2
                          ),

    'inh_decision': pynnx.Pop(1, neuron_class,
                              decision_parameters, label='Inh Decision Neuron',
                              hicann=hicanns['exciter'],
                              gmax=1023
                              ),

    'inh_kenyon': pynnx.Pop(1, neuron_class,
                            decision_parameters, label='Inh Kenyon Neuron',
                            hicann=hicanns['feedback'],
                            gmax=1023
                            ),

    ### make neurons spike right before a new pattern is shown
    # 'tick': pynnx.Pop(1, 'SpikeSourceArray',
                      # {'spike_times': tick_spikes}, label='Tick Neurons',
                      # hicann_id=hicanns['tick']),
    # 'feedback': pynnx.Pop(args.nDN, neuron_class,
                          # fb_parameters, label='Feedback Neurons',
                          # hicann_id=hicanns['feedback']),

    ### add current if neuron hasn't spiked yet
    # 'exciter': pynnx.Pop(args.nDN, neuron_class,
                         # exciter_parameters, label='Threshold Reducer',
                         # hicann_id=hicanns['exciter']),

    # 'exciter src': pynnx.Pop(args.nDN, 'SpikeSourcePoisson',
                             # {'rate': 1000.0, 'start': start_dt,
                             # 'duration': n_exciter_samples * sample_dt},
                             # label='Threshold Reducer Source',
                             # hicann_id=hicanns['exciter src']),

}

for i in range(div_kc):
    kpop = 'kenyon%d'%i
    populations[kpop] = pynnx.Pop(nkc, neuron_class,
                            kenyon_parameters, label='Kenyon Cell %d'%i,
                            hicann=hicanns['kenyon'][i],
                            gmax=2
                            )
    pynnx.set_recording(populations[kpop], 'spikes')

pynnx.set_recording(populations['decision'], 'spikes')
np.random.seed()
# populations['decision'].initialize(v=np.random.uniform(-120.0, -50.0, size=args.nDN))

if record_all:
    pynnx.set_recording(populations['horn'], 'spikes')
    pynnx.set_recording(populations['feedback'], 'spikes')
    pynnx.set_recording(populations['decision'], 'v')
    pynnx.set_recording(populations['kenyon'], 'v')
    pynnx.set_recording(populations['exciter'], 'spikes')

sys.stdout.write('Creating projections\n')
sys.stdout.flush()

static_w = {
    'AL to KC': W2S * 1.0 * (100.0 / float(args.nAL)),
    'KC to KC': W2S * (1.0 * (2500.0 / float(args.nKC))),

    'KC to DN': W2S * (0.01 * (2500.0 / float(args.nKC))),
    'DN to DN': W2S * (5.0 * (100.0 / float(args.nDN))),

    'DN to FB': W2S * (0.75 * (100.0 / float(args.nDN))),
    'FB to DN': W2S * (1.5 * (100.0 / float(args.nDN))),
    'TK to FB': W2S * (0.75 * (100.0 / float(args.nDN))),

    'DN to TR':  W2S * (5.0 * (100.0 / float(args.nDN))),
    'TRS to TR': W2S * (1.0 * (100.0 / float(args.nDN))),
    'TR to DN':  W2S * (0.2 * (100.0 / float(args.nDN))),
    # 'DN to TR':  W2S * (1.0 * (100.0 / float(args.nDN))),
    # 'TRS to TR': W2S * (0.000001 * (100.0 / float(args.nDN))),
    # 'TR to DN':  W2S * (0.00000001 * (100.0 / float(args.nDN))),
    
    'INH': 1.0,
    'EXC': 1.0,
}
pprint(static_w)

rand_w = {
    'AL to KC': static_w['AL to KC'],
}

w_max = (static_w['KC to DN'] * 1.0) * 1.2
# w_min = -5. * w_max
w_min = 0.0
print("\nw_min = {}\tw_max = {}\n".format(w_min, w_max))

gain_list = []

out_lists = [output_connection_list(nkc, args.nDN, args.probKC2DN,
                                   static_w['KC to DN'],
                                   args.inactiveScale,
                                   delay=3,
                                   seed=None,
                                   clip_to=w_max
                                   ) for _ in range(div_kc)]

out_neighbours = []

t_plus = 5.0
t_minus = 15.0
a_plus = 0.1
a_minus = 0.1


stdp = {
    'timing_dependence': {
        'name': 'SpikePairRule',
        'params': {'tau_plus': t_plus,
                   'tau_minus': t_minus,
                   },
    },
    'weight_dependence': {
        'name': 'AdditiveWeightDependence',
        'params': {
            'w_min': w_min,
            'w_max': w_max,
            'A_plus': a_plus, 'A_minus': a_minus,
        },
    }
}

projections = {
     ### Inhibitory feedback --- decision neurons
    'DN to IDN': pynnx.Proj(populations['decision'], populations['inh_decision'],
                           'AllToAllConnector', weights=static_w['EXC'], delays=timestep,
                           conn_params={'allow_self_connections': True}, target='excitatory', label='DN to IDN',
                           digital_weights=15
                           ),

    'IDN to DN': pynnx.Proj(populations['inh_decision'], populations['decision'],
                           'AllToAllConnector', weights=static_w['INH'], delays=timestep,
                           conn_params={'allow_self_connections': True}, target='inhibitory', label='IDN to DN',
                           digital_weights=15
                           ),

    ### make decision spike just before the next pattern to reduce weights corresponding to that input
    # 'DN to FB': pynnx.Proj(populations['decision'], populations['feedback'],
                           # 'OneToOneConnector', weights=static_w['DN to FB'], delays=15.0,
                           # target='excitatory', label='DN to FB'),

    # 'FB to DN': pynnx.Proj(populations['feedback'], populations['decision'],
                           # 'OneToOneConnector', weights=static_w['FB to DN'], delays=15.0,
                           # target='excitatory', label='FB to DN'),

    # 'TK to FB': pynnx.Proj(populations['tick'], populations['feedback'],
                           # 'AllToAllConnector', weights=static_w['TK to FB'], delays=1.0,
                           # target='excitatory', label='TK to FB'),

    ### have some more current comming into decicions if they have not spiked recently
    # 'TR to DN': pynnx.Proj(populations['exciter'], populations['decision'],
                           # 'OneToOneConnector', weights=static_w['TR to DN'], delays=1.0,
                           # target='excitatory', label='TR to DN'),

    # 'DN to TR': pynnx.Proj(populations['decision'], populations['exciter'],
                           # 'OneToOneConnector', weights=static_w['DN to TR'], delays=timestep,
                           # target='inhibitory', label='DN to TR'),

    # 'TRS to TR': pynnx.Proj(populations['exciter src'], populations['exciter'],
                           # 'OneToOneConnector', weights=static_w['TRS to TR'], delays=1.0,
                           # target='excitatory', label='TRS to TR'),

}
for i in range(div_kc):
    kAL2KC = 'AL to KC_%d'%i
    kpop = 'kenyon%d'%i
    projections[kAL2KC] = pynnx.Proj(populations['antenna'], populations[kpop],
                                'FixedProbabilityConnector', weights=rand_w['AL to KC'], delays=4.0,
                                conn_params={'p_connect': args.probAL2KC}, label=kAL2KC,
                                digital_weights=8
                                )

    kKC2DN = 'KC_%d to DN'%i
    projections[kKC2DN] = pynnx.Proj(populations[kpop], populations['decision'],
                                weights=None, delays=None,
                                conn_class='FromListConnector', 
                                conn_params={'conn_list': out_lists[i]}, 
                                # conn_class='FixedProbabilityConnector', 
                                # conn_params={'p_connect': 0.1}, 
                                label=kKC2DN,
                                #stdp=stdp,
                                digital_weights=1
                                )

    kKC2IKC = 'KC_%d to IKC'%i
    projections[kKC2IKC] = pynnx.Proj(populations[kpop], populations['inh_kenyon'],
                            'AllToAllConnector', weights=static_w['EXC'], 
                            delays=timestep,
                            conn_params={'allow_self_connections': True}, 
                            target='excitatory', label=kKC2IKC)
                            
    kIKC2KC = 'IKC to KC_%d'%i
    projections[kIKC2KC] = pynnx.Proj(populations['inh_kenyon'], populations[kpop],
                            'AllToAllConnector', weights=static_w['INH'], 
                            delays=timestep,
                            conn_params={'allow_self_connections': True}, 
                            target='inhibitory', label=kIKC2KC)


sys.stdout.write('Running simulation\n')
sys.stdout.flush()


sweights = []
for out_list in out_lists:
    starting_weights = np.zeros((nkc, args.nDN))
    for i, j, v, d in out_list:
        starting_weights[int(i), int(j)] = v
    starting_weights = starting_weights.flatten()
    sweights.append(starting_weights)

weights = [sweights]


if fixed_loops == 0:
    # weight_sample_dt = 10.
    weight_sample_dt = float(sample_dt * (args.nSamplesAL * 0.1))
    n_loops = np.ceil(sim_time / weight_sample_dt)
else:
    n_loops = fixed_loops
    weight_sample_dt = np.ceil(sim_time / float(n_loops))

print("num loops = {}\ttime per loop {}".format(n_loops, weight_sample_dt))
now = datetime.datetime.now()
sys.stdout.write("\tstarting time is {:02d}:{:02d}:{:02d}\n".format(now.hour, now.minute, now.second))
sys.stdout.flush()

tmp_w = []
t0 = time.time()
for loop in np.arange(n_loops):
    sys.stdout.write("\trunning loop {} of {}\t".format(loop + 1, n_loops))
    sys.stdout.flush()

    loop_t0 = time.time()
    now = datetime.datetime.now()
    sys.stdout.write("starting {:02d}:{:02d}:{:02d}\t".format(now.hour, now.minute, now.second))
    sys.stdout.flush()


    pynnx.run(weight_sample_dt)


    secs_to_run = time.time() - loop_t0
    mins_to_run = secs_to_run // 60
    secs_to_run -= mins_to_run * 60
    hours_to_run = mins_to_run // 60
    mins_to_run -= hours_to_run * 60
    secs_to_run, mins_to_run, hours_to_run = int(secs_to_run), int(mins_to_run), int(hours_to_run)

    sys.stdout.write('lasted {:02d}h: {:02d}m: {:02d}s\n'.format(hours_to_run, mins_to_run, secs_to_run))
    sys.stdout.flush()
    tmp_w[:] = []
    for k in sorted(projections.keys()):
        if k.startswith('KC') and k.endswith('to DN'):
            tmp_w.append(pynnx.get_weights(projections[k]).flatten())
    
    # print(loop, tmp_w.shape)
    weights.append(tmp_w)

post_horn = []
secs_to_run = time.time() - t0

mins_to_run = secs_to_run // 60
secs_to_run -= mins_to_run * 60
hours_to_run = mins_to_run // 60
mins_to_run -= hours_to_run * 60
secs_to_run, mins_to_run, hours_to_run = int(secs_to_run), int(mins_to_run), int(hours_to_run)

sys.stdout.write('\n\nDone!\tRunning simulation - lasted {:02d}h: {:02d}m: {:02d}s\n\n'. \
                 format(hours_to_run, mins_to_run, secs_to_run))
sys.stdout.flush()

sys.stdout.write('Getting spikes:\n')
sys.stdout.flush()

sys.stdout.write('\tKenyon\n')
sys.stdout.flush()
k_spikes = [pynnx.get_record(populations[k], 'spikes') \
            for k in populations if k.startswith('kenyon')]

sys.stdout.write('\tDecision\n')
sys.stdout.flush()
out_spikes = pynnx.get_record(populations['decision'], 'spikes')

out_high = get_high_spiking(out_spikes, 0, weight_sample_dt, 5)


if record_all:
    # sys.stdout.write('\tHorn\n')
    # sys.stdout.flush()
    # horn_spikes = pynnx.get_record(populations['horn'], 'spikes')

    # sys.stdout.write('\tFeedback\n')
    # sys.stdout.flush()
    # fb_spikes = pynnx.get_record(populations['feedback'], 'spikes')
    horn_spikes = [[]]
    fb_spikes = [[]]

    sys.stdout.write('\tExciter\n')
    sys.stdout.flush()
    exciter_spikes = pynnx.get_record(populations['exciter'], 'spikes')

else:
    horn_spikes = [[]]
    fb_spikes = [[]]
    exciter_spikes = [[]]

sys.stdout.write('Done!\tGetting spikes\n\n')
sys.stdout.flush()


if record_all:
    sys.stdout.write('Getting voltages\n')
    sys.stdout.flush()

    dn_voltage = pynnx.get_record(populations['decision'], 'v')
    kc_voltage = pynnx.get_record(populations['kenyon'], 'v')

    sys.stdout.write('Done!\tGetting voltages\n\n')
    sys.stdout.flush()

else:
    dn_voltage = [np.array([[0, 0]])]
    kc_voltage = [np.array([[0, 0]])]

sys.stdout.write('Getting weights:\n')
sys.stdout.flush()
sys.stdout.write('\tKenyon\n')
sys.stdout.flush()
# try:
final_weights = weights[-1]

for k in projections:
    if k.startswith('AL to KC'):
        wwwtmp = pynnx.get_weights(projections[k])
        non_nan = np.where(~ np.isnan(wwwtmp))[0].size
        print(float(non_nan)/float(wwwtmp.size))


sys.stdout.write('Done!\t Getting weights\n\n')
sys.stdout.flush()

pynnx.end()

sys.stdout.write('Saving experiment\n')
sys.stdout.flush()
# fname = 'mbody-'+args_to_str(args)+'.npz'
fname = 'bss-mbody-experiment-1.npz'
np.savez_compressed(fname, args=args, sim_time=sim_time,
                    input_spikes=spike_times, input_vectors=input_vecs,
                    input_samples=samples, sample_indices=sample_indices,
                    output_start_connections=out_lists, 
                    lateral_horn_connections=gain_list,
                    output_end_weights=final_weights, 
                    static_weights=static_w, stdp_params=stdp,
                    kenyon_spikes=k_spikes, decision_spikes=out_spikes, 
                    horn_spikes=horn_spikes,
                    neuron_parameters=neuron_params,
                    sample_dt=sample_dt, start_dt=start_dt, max_rand_dt=max_rand_dt,
                    dn_voltage=dn_voltage, kc_voltage=kc_voltage,
                    high_dt=high_dt,
                    low_freq=low_freq, high_freq=high_freq,
                    weights=weights, weight_sample_dt=weight_sample_dt,
                    timestep=timestep,
                    post_horn_weights=post_horn,
                    tick_spikes=tick_spikes,
                    fb_spikes=fb_spikes,
                    exciter_spikes=exciter_spikes,
                    n_test_samples=n_test_samples,
                    )
sys.stdout.write('Done!\tSaving experiment\n\n')
sys.stdout.flush()

if args.renderSpikes:
    render_spikes(k_spikes, 'Kenyon activity', 'kenyon_activity.pdf')

    render_spikes(out_spikes, 'Output activity', 'output_activity.pdf')
