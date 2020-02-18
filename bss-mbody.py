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
import copy

# from spikevo import *
# from spikevo.pynn_transforms import PyNNAL
# from spikevo.wafer import Wafer as WAL
import argparse
from pprint import pprint
from args_setup import get_args
from input_utils import *

from pyhalbe import HICANN
import pyhalbe.Coordinate as C
from pymarocco import PyMarocco
from pymarocco.runtime import Runtime
from pymarocco.coordinates import LogicalNeuron
from pymarocco.results import Marocco
from pymarocco import Defects
import pysthal
from pysthal.command_line_util import init_logger
import pyhmf as sim

from bss_utils import *
from bss_utils import _BSS_set_hicann_sthal_params
import bss_utils
print(dir(bss_utils))

init_logger("WARN", [
    ("guidebook", "INFO"),
    ("marocco", "INFO"),
    ("Calibtic", "INFO"),
    ("sthal", "INFO")
])




args = get_args()
pprint(args)

backend = args.backend

neuron_class = sim.IF_cond_exp
# heidelberg's brainscales seems to like these params

e_rev = 92  # mV
# e_rev = 500.0 #mV

base_params = {
    #3 'cm': 0.1,  # nF
    'cm': 0.2,  # nF
    'v_reset': -70.,  # mV
    'v_rest': -65.,  # mV
    'v_thresh': -50.,  # mV
    #3 'v_thresh': -50.,  # mV
    #3 'e_rev_I': -e_rev, #mV
    #3 'e_rev_E': 0.,#e_rev, #mV
    #3 'tau_m': 10.,  # ms
    'tau_m': 10.,  # ms
    'tau_refrac': 1.,  # ms
    'tau_syn_E': 1.0,  # ms
    'tau_syn_I': 5.0,  # ms
    
    # 'cm': 0.01,
    # 'v_reset': -20.,
    # 'v_rest': -20.,
    # 'v_thresh': -10,
    # 'e_rev_I': -100.,
    # 'e_rev_E': 60.,
    # 'tau_m': 20.,
    # 'tau_refrac': 0.1,
    # 'tau_syn_E': 5.,
    # 'tau_syn_I': 5.,
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
sample_dt, start_dt, max_rand_dt = 50, 5, 0.
sim_time = sample_dt * args.nSamplesAL * args.nPatternsAL
timestep = 0.1
regenerate = args.regenerateSamples
record_all = args.recordAllOutputs and args.nSamplesAL <= 50
fixed_loops = args.fixedNumLoops
n_explore_samples = min(args.nPatternsAL * 10, np.round(args.nSamplesAL * args.nPatternsAL * 0.01))
n_exciter_samples = min(args.nPatternsAL * 100, np.round(args.nSamplesAL * args.nPatternsAL * 0.1))
n_test = np.round(args.nSamplesAL * 1.0/10.0)
n_test_samples = min(1000, n_test * args.nPatternsAL)
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

samples = generate_samples(input_vecs, args.nSamplesAL, n_test, args.probNoiseSamplesAL, seed=234,
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
    sample_indices, spike_times = samples_to_spike_times(samples, args.nSamplesAL, n_test,
                                    sample_dt, start_dt, max_rand_dt, timestep,
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
### ======================   SIM INIT   ========================== ###

sys.stdout.write('Creating simulator abstraction\n')
sys.stdout.flush()

wafer = int(os.environ.get("WAFER", 33))
marocco = PyMarocco()
marocco.backend = PyMarocco.Hardware

marocco.default_wafer = C.Wafer(wafer)
runtime = Runtime(marocco.default_wafer)

# calib_path = "/wang/data/calibration/brainscales/WIP-2018-09-18"
# marocco.calib_path = calib_path
# marocco.defects.path = marocco.calib_path

marocco.verification = PyMarocco.Skip
marocco.checkl1locking = PyMarocco.SkipCheck
marocco.continue_despite_synapse_loss = True



sim.setup(timestep=1.0, min_delay=1.0, marocco=marocco, marocco_runtime=runtime)


sys.stdout.write('Done!\tCreating simulator abstraction\n\n')
sys.stdout.flush()

sys.stdout.write('Creating populations\n')
sys.stdout.flush()

#######################################################################
#######################################################################
#######################################################################

div_kc = 5
div_kc = 10
# div_kc = 15
central_hicann = 76
# central_hicann = 107
# central_hicann = 171
# central_hicann = 283
# central_hicann = 275
hicanns = get_hicanns(central_hicann, div_kc, seed=args.hicann_seed, 
            max_dist=5, n_per_pop=5, manual=bool(0))
# pprint(hicanns)
nkc = int(np.ceil(args.nKC/float(div_kc)))
print("\n\nnumber of neurons in per kenyon subpop = {}\n".format(nkc))

#######################################################################
#######################################################################
#######################################################################
#######################################################################

populations = {
    'antenna': [
        sim.Population(1, sim.SpikeSourceArray,
            {'spike_times': spike_times[i]}, label='Antennae Lobe',)
        for i in range(args.nAL)
    ],
    'decision': sim.Population(args.nDN, neuron_class,
                          decision_parameters, label='Decision Neurons',),

    'inh_decision': sim.Population(1, neuron_class,
                        decision_parameters, label='Inh Decision Neuron',),

    'inh_kenyon': sim.Population(1, neuron_class,
                    decision_parameters, label='Inh Kenyon Neuron',),

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

i2t = lambda x: "%02d"%x
for i in range(div_kc):
    kpop = 'kenyon_%s'%(i2t(i))
    populations[kpop] = sim.Population(nkc, neuron_class,
                            kenyon_parameters, label='Kenyon Cell %d'%i,)
    populations[kpop].record()

populations['decision'].record()
populations['inh_decision'].record()
populations['inh_kenyon'].record()
np.random.seed()
# populations['decision'].initialize(v=np.random.uniform(-120.0, -50.0, size=args.nDN))


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
print("args.probAL2KC = {}".format(args.probAL2KC))
print("args.probKC2DN = {}".format(args.probKC2DN))

gain_list = []
in_lists = [input_connection_list(args.nAL, nkc, args.probAL2KC, rand_w['AL to KC']) \
                for _ in range(div_kc)]

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


conn_lists = {
    'DN to IDN': all_to_all(
        populations['decision'], populations['inh_decision'],
        weights=static_w['EXC'], delays=timestep),
        
    'IDN to DN': all_to_all(
        populations['inh_decision'], populations['decision'],
        weights=static_w['INH'], delays=timestep),
}

weight_matrices = {}
projections = {
     ### Inhibitory feedback --- decision neurons
    'DN to IDN': sim.Projection(
        populations['decision'], populations['inh_decision'],
        sim.FromListConnector(conn_lists['DN to IDN']), 
        target='excitatory', label='DN to IDN',),

    'IDN to DN': sim.Projection(
        populations['inh_decision'], populations['decision'],
        sim.FromListConnector(conn_lists['IDN to DN']), 
        target='inhibitory', label='IDN to DN',),

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

weight_matrices['DN to IDN'] = list_to_matrix(
    populations['decision'], populations['inh_decision'],
    conn_lists['DN to IDN'])

weight_matrices['IDN to DN'] = list_to_matrix(
    populations['inh_decision'], populations['decision'],
    conn_lists['IDN to DN'])


for i in range(div_kc):
    kAL2KC = 'AL to KC_%s'%(i2t(i))
    kpop = 'kenyon_%s'%(i2t(i))
    # projections[kAL2KC] = pynnx.Proj(populations['antenna'], populations[kpop],
    #                             'FixedProbabilityConnector', weights=rand_w['AL to KC'], delays=4.0,
    #                             conn_params={'p_connect': args.probAL2KC}, label=kAL2KC,
    #                             # digital_weights=1
    #                             )
    
    projections[kAL2KC] = []
    weight_matrices[kAL2KC] = [None for _ in range(args.nAL)]
    for pre_id, _in_conn_list in enumerate(in_lists[i]):
        print("\t{} to {}".format(pre_id, i))
        print("\t{}\tto \n\t{}".format(populations['antenna'][pre_id], populations[kpop]))
        for r in _in_conn_list:
            print("\t\t{}".format(r))

        projections[kAL2KC].append(sim.Projection(
            populations['antenna'][pre_id], populations[kpop],
            sim.FromListConnector(_in_conn_list), 
            label=kAL2KC, target='excitatory',
        ))
        
        weight_matrices[kAL2KC][pre_id] = list_to_matrix(
            populations['antenna'][pre_id], populations[kpop],
            _in_conn_list
        )
    
    kKC2DN = 'KC_%s to DN'%(i2t(i))
    projections[kKC2DN] = sim.Projection(
        populations[kpop], populations['decision'],
        sim.FromListConnector(out_lists[i]), 
        label=kKC2DN, target='excitatory',
    )
    
    weight_matrices[kKC2DN] = list_to_matrix(
        populations[kpop], populations['decision'],
        out_lists[i]
    )

    # kKC2IKC = 'KC_%s to IKC'%(i2t(i))
    # projections[kKC2IKC] = pynnx.Proj(populations[kpop], populations['inh_kenyon'],
                            # 'FromListConnector', weights=None, delays=None,
                            # conn_params={'conn_list': 
                                # all_to_all(populations[kpop], populations['inh_kenyon'],
                                           # weights=static_w['EXC'], delays=timestep)}, 
                            # target='excitatory', label=kKC2IKC,
                            # digital_weights=15,
                            # )
                            
    # kIKC2KC = 'IKC to KC_%s'%(i2t(i))
    # projections[kIKC2KC] = pynnx.Proj(populations['inh_kenyon'], populations[kpop],
                            # 'FromListConnector', weights=None, 
                            # delays=None,
                            # conn_params={'conn_list': 
                                # all_to_all(populations['inh_kenyon'], populations[kpop],
                                           # weights=static_w['INH'], delays=timestep)}, 
                            # target='inhibitory', label=kIKC2KC,
                            # digital_weights=15,
                            # )


sweights = []
for out_list in out_lists:
    starting_weights = np.zeros((nkc, args.nDN))
    for i, j, v, d in out_list:
        starting_weights[int(i), int(j)] = v
    starting_weights = starting_weights#.flatten()
    sweights.append(starting_weights)

weights = [sweights]


if fixed_loops == 0:
    # weight_sample_dt = 10.
    weight_sample_dt = float(sample_dt * (args.nSamplesAL * 0.1))
    n_loops = np.ceil(sim_time / weight_sample_dt)
else:
    n_loops = fixed_loops
    weight_sample_dt = np.ceil(sim_time / float(n_loops))

total_t = sample_dt * args.nSamplesAL * args.nPatternsAL
weight_sample_dt = total_t
noise_count_threshold = max(0, int(args.nSamplesAL * 0.0))
noise_count_threshold_out = args.nSamplesAL * 0.8
n_loops = 5
base_train_loops = max(n_loops, 10)



### ====================== PERFORM MAPPING =========================== ###



sys.stdout.write('\n\n\nMaping simulation\n\n\n')
sys.stdout.flush()

seed = 0
marocco.l1_routing.shuffle_switches_seed(seed)

marocco.skip_mapping = False
marocco.backend = PyMarocco.None

sim.reset()
sim.run(weight_sample_dt)



### ===================   SET HICANN PARAMS   ======================== ###


wafer = runtime.wafer()
hicanns_in_use = wafer.getAllocatedHicannCoordinates()

for hicann in hicanns_in_use:
    _BSS_set_hicann_sthal_params(wafer, hicann, 1023)


marocco.skip_mapping = True
marocco.backend = PyMarocco.Hardware
# Full configuration during first step
marocco.hicann_configurator = pysthal.ParallelHICANNv4Configurator()



### ======================   HARDWARE RUNS   ========================= ###




sys.stdout.write('Running simulation\n')
sys.stdout.flush()

print("num loops = {}\ttime per loop {}".format(n_loops, weight_sample_dt))
now = datetime.datetime.now()
sys.stdout.write(
    "\tstarting time is {:02d}:{:02d}:{:02d}\n".format(now.hour, now.minute, now.second))
sys.stdout.flush()
k_spikes = []
ik_spikes = []
out_spikes = []
iout_spikes = []
w_list = []
tmp_w = {}
t0 = time.time()
for loop in np.arange(n_loops):

    sys.stdout.write("\n\n\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    sys.stdout.write("\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    sys.stdout.write("\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    sys.stdout.flush()
    
    sys.stdout.write("\n\trunning loop {} of {}\t".format(loop + 1, n_loops))
    sys.stdout.flush()

    loop_t0 = time.time()
    now = datetime.datetime.now()
    sys.stdout.write("starting {:02d}:{:02d}:{:02d}\n\n".format(now.hour, now.minute, now.second))
    sys.stdout.flush()

    ### ---------------------------------
    ### run experiment 
    for k in projections:
        proj = projections[k]
        ws = weight_matrices[k]
        print(k)
        if 'AL to KC' in k:
            print("proj is a list")
            for proj_idx, local_proj in enumerate(proj):
                set_digital_weights(ws[proj_idx], local_proj, runtime)
                
        else:
            set_digital_weights(ws, proj, runtime)
                
    sim.run(weight_sample_dt) 

    f = open('it_ran_log.txt', 'a+')
    f.write(u'%s\n'%(args.hicann_seed))
    f.close()

    secs_to_run = time.time() - loop_t0

    sys.stdout.write('lasted {}\n'.format(secs_to_hms(secs_to_run)))
    sys.stdout.flush()

    # print(loop, tmp_w.shape)
    weights.append(weight_matrices.copy())
    
    ### ---------------------------------
    ### grab spikes to do further learning (blacklist and structural plasticity)
    sys.stdout.write('\n\n\nGetting spikes:\n')
    sys.stdout.flush()

    sys.stdout.write('\tKenyon\n')
    sys.stdout.flush()
    _k_spikes = {k: populations[k].getSpikes()
                    for k in populations if k.lower().startswith('kenyon')}
    bk_spikes = {k: bin_spikes_per_sample(0, weight_sample_dt, sample_dt, _k_spikes[k]) \
                 for k in populations if k.lower().startswith('kenyon')}
    
    for k in _k_spikes:
        ksum = 0
        for times in _k_spikes[k]:
            ksum += len(times)
        print("\n%s sum = %s"%(k, ksum))

    sys.stdout.write('\tDecision\n')
    sys.stdout.flush()
    _out_spikes = populations['decision'].getSpikes()
    bout_spikes = bin_spikes_per_sample(0, weight_sample_dt, sample_dt, _out_spikes)
    osum = 0
    for times in _out_spikes:
        osum += len(times)
    print("\n%s sum = %s"%('output', osum))


    _ik_spikes = populations['inh_kenyon'].getSpikes()
    _io_spikes = populations['inh_decision'].getSpikes()



    ### ---------------------------------
    ### get highest spiking neurons, 
    k_high = {\
        k: get_high_spiking(_k_spikes[k], 0, weight_sample_dt, noise_count_threshold) \
                  for k in _k_spikes if k.lower().startswith('kenyon')}
    
    print("k_high")
    print(k_high)
    
    out_high = get_high_spiking(_out_spikes, 0, weight_sample_dt, noise_count_threshold_out)
    
    w_list.append(copy.deepcopy(weight_matrices))

    ### Update blacklists
    ### reduce in-K weights
    for k in sorted(k_high.keys()):
        int_idx = int(k.split("_")[-1])
        w_idx = "AL to KC_%s"%(i2t(int_idx))
        pynnx_k = "Kenyon Cell %s"%(i2t(int_idx))
        # if loop < base_train_loops:
        if loop < 20:
            neuron_ids = k_high[k]
            n_to_delete = 10
            for pre_idx, ws in enumerate(weight_matrices[w_idx]):
                weight_matrices[w_idx][pre_idx][:] = \
                    reduce_influence(neuron_ids, ws, 'post', n_to_delete)
            
                

    ### reduce K-iK weights
    ### reduce D-iD weights
    
    # Kenyon Cell %d
    # Decision 
    # for k in sorted(bk_spikes.keys()):
        # print(k)
        # int_idx = int(k.split("_")[-1])
        # print(int_idx)
        # w_idx = "KC_%s to DN"%(i2t(int_idx))
        # pynnx_k = "Kenyon Cell %s"%(i2t(int_idx))
        # print(k, int_idx, w_idx, pynnx_k)
        # print(tmp_w[w_idx].shape)
        # print("Decision Neurons")
        # print(out_high.keys())
        
        # if loop > base_train_loops:
            # projections[w_idx]._PyNNAL__weight_matrix[:] = \
                # structural_plasticity(bk_spikes[k], bout_spikes, tmp_w[w_idx],
                    # {}, {})


            # projections[w_idx]._PyNNAL__weight_matrix[:] = \
                # reduce_influence(k_high[k], projections[w_idx], PRE)


    k_spikes.append(_k_spikes)
    ik_spikes.append(_ik_spikes)
    out_spikes.append(_out_spikes)
    iout_spikes.append(_io_spikes)
    # w_list.append(copy.deepcopy(weight_matrices))
    np.savez_compressed("experiment_at_loop_%03d.npz"%loop,
        kenyon_spikes=k_spikes, ik_spikes=ik_spikes, 
        decision_spikes=out_spikes, iout_spikes=iout_spikes,
        weights=weights, args=args, sim_time=weight_sample_dt,
        sample_dt=sample_dt,
        input_spikes=spike_times, input_vectors=input_vecs,
        input_samples=samples, sample_indices=sample_indices,
        n_test_samples=n_test_samples,
        weight_matrices=weight_matrices,
        w_list=w_list,
    )
    
    print("-----------------------------------------------------------")
    print("-----------------------------------------------------------")
    print("-----------------------------------------------------------")
    print("-----------------------------------------------------------")
    print("-----------------------------------------------------------")
    print("-----------------------------------------------------------")
    print("-----------------------------------------------------------")
    
w_list.append(copy.deepcopy(weight_matrices))

post_horn = []
secs_to_run = time.time() - t0

sys.stdout.write('\n\nDone!\tRunning simulation - lasted {}\n'.format(secs_to_hms(secs_to_run)))
sys.stdout.flush()



horn_spikes = [[]]
fb_spikes = [[]]
exciter_spikes = [[]]

sys.stdout.write('Done!\tGetting spikes\n\n')
sys.stdout.flush()


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
        for p in projections[k]:
            wwwtmp = p.getWeights()
            non_nan = np.where(~ np.isnan(wwwtmp))[0].size
            # print(float(non_nan)/float(wwwtmp.size))
        

sys.stdout.write('Done!\t Getting weights\n\n')
sys.stdout.flush()

sim.end()

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
                    inh_kenyon_spikes=ik_spikes, inh_decision_spikes=iout_spikes, 
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
                    w_list=w_list,
                    )
sys.stdout.write('Done!\tSaving experiment\n\n')
sys.stdout.flush()

if args.renderSpikes:
    render_spikes(k_spikes, 'Kenyon activity', 'kenyon_activity.pdf')

    render_spikes(out_spikes, 'Output activity', 'output_activity.pdf')
