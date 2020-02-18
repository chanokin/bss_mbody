import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import copy
import os

from pyhalbe import HICANN
import pyhalbe.Coordinate as C
from pymarocco import PyMarocco, Defects
from pymarocco.runtime import Runtime
from pymarocco.coordinates import LogicalNeuron
from pymarocco.results import Marocco
from pymarocco import Defects
import pysthal
from pysthal.command_line_util import init_logger
import pyhmf as sim

init_logger("WARN", [
    ("guidebook", "INFO"),
    ("marocco", "INFO"),
    ("Calibtic", "INFO"),
    ("sthal", "INFO")
])


def _BSS_set_hicann_sthal_params(_wafer, _hicann, gmax, gmax_div=1):
    fgs = _wafer[_hicann].floating_gates

    # set parameters influencing the synaptic strength
    for block in C.iter_all(C.FGBlockOnHICANN):
        fgs.setShared(block, HICANN.shared_parameter.V_gmax0, gmax)
        fgs.setShared(block, HICANN.shared_parameter.V_gmax1, gmax)
        fgs.setShared(block, HICANN.shared_parameter.V_gmax2, gmax)
        fgs.setShared(block, HICANN.shared_parameter.V_gmax3, gmax)

    for driver in C.iter_all(C.SynapseDriverOnHICANN):
        for row in C.iter_all(C.RowOnSynapseDriver):
            _wafer[_hicann].synapses[driver][row].set_gmax_div(
                C.left, gmax_div)
            _wafer[_hicann].synapses[driver][row].set_gmax_div(
                C.right, gmax_div)

    # don't change values below
    for ii in range(fgs.getNoProgrammingPasses()):
        cfg = fgs.getFGConfig(C.Enum(ii))
        cfg.fg_biasn = 0
        cfg.fg_bias = 0
        fgs.setFGConfig(C.Enum(ii), cfg)

    for block in C.iter_all(C.FGBlockOnHICANN):
        fgs.setShared(block, HICANN.shared_parameter.V_dllres, 275)
        fgs.setShared(block, HICANN.shared_parameter.V_ccas, 800)


def _BSS_set_sthal_params(runtime, gmax=1023, gmax_div=1):
    """
    synaptic strength:
    gmax: 0 - 1023, strongest: 1023
    gmax_div: 1 - 15, strongest: 1
    """
    _wafer = runtime.wafer()
    # for all HICANNs in use
    for _hicann in wafer.getAllocatedHicannCoordinates():
        _BSS_set_hicann_sthal_params(_wafer, _hicann, gmax, gmax_div)


def list_conn(n_pre, n_post, prob, w, d):
    return [(pre, post, w, d) for pre in range(n_pre) for post in range(n_post) \
                if np.random.uniform() <= prob]

def set_digital_weights(in_mtx, projections, total_in, total_out, bss_runtime, digital_w=15):
    n_per_out = total_out // len(projections[0])

    original_decoders = {}
    for i, prjs in enumerate(projections):
        for j, p in enumerate(prjs):
            rtime_res = bss_runtime.results()
            synapses = rtime_res.synapse_routing.synapses()
            proj_items = synapses.find(p)

            for _item in proj_items:
                syn = _item.hardware_synapse()
                pre = _item.source_neuron().neuron_index()
                post = _item.target_neuron().neuron_index()

                post += j * n_per_out
                w = in_mtx[pre, post]

                proxy = runtime.wafer()[syn.toHICANNOnWafer()].synapses[syn]
                if syn not in original_decoders:
                    original_decoders[syn] = copy.copy(proxy.decoder)

                if np.isnan(w):
                    proxy.weight = HICANN.SynapseWeight(0)
                    ### SETTING SYNAPSE TO DISABLED DECODER, DISABLING SYNAPSE
                    proxy.decoder = SYNAPSE_DECODER_DISABLED_SYNAPSE
                elif w <= 0.0:
                    proxy.weight = HICANN.SynapseWeight(0)
                    proxy.decoder = original_decoders[syn]
                else:
                    proxy.weight = HICANN.SynapseWeight(digital_w)
                    proxy.decoder = original_decoders[syn]



def zero_digital_weights(projections, total_in, total_out, bss_runtime, digital_w=15):
    n_per_out = total_out // len(projections[0])
    in_mtx = np.zeros((total_in, total_out))
    for i, prjs in enumerate(projections):
        for j, p in enumerate(prjs):
            c0 = j * n_per_out
            c1 = c0 + n_per_out
            in_mtx[i, c0:c1] = p.getWeights(format='array')
    out_mtx = in_mtx.copy()

    whr = np.where(~np.isnan(out_mtx))
    zeros = np.where(np.random.uniform(0.0, 1.0, size=whr[0].shape) <= 1.0)
    rows, cols = whr[0][zeros], whr[1][zeros]
    out_mtx[rows, cols] = 0

    original_decoders = {}
    for i, prjs in enumerate(projections):
        for j, p in enumerate(prjs):
            rtime_res = bss_runtime.results()
            synapses = rtime_res.synapse_routing.synapses()
            proj_items = synapses.find(p)

            for _item in proj_items:
                syn = _item.hardware_synapse()
                pre = _item.source_neuron().neuron_index()
                post = _item.target_neuron().neuron_index()

                post += j * n_per_out
                w = out_mtx[pre, post]

                proxy = runtime.wafer()[syn.toHICANNOnWafer()].synapses[syn]
                if syn not in original_decoders:
                    original_decoders[syn] = copy.copy(proxy.decoder)

                if np.isnan(w) or w <= 0.0:
                    proxy.weight = HICANN.SynapseWeight(0)
                    ### SETTING SYNAPSE TO DISABLED DECODER, DISABLING SYNAPSE
                    proxy.decoder = SYNAPSE_DECODER_DISABLED_SYNAPSE
                else:
                    proxy.weight = HICANN.SynapseWeight(digital_w)
                    proxy.decoder = original_decoders[syn]

    return out_mtx

############################################################################
############################################################################
############################################################################

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

SYNAPSE_DECODER_DISABLED_SYNAPSE = HICANN.SynapseDecoder(1)

### ====================== NETWORK CONSTRUCTION =========================== ###
sim.setup(timestep=1.0, min_delay=1.0, marocco=marocco, marocco_runtime=runtime)

e_rev = 92  # mV
# e_rev = 500.0 #mV

base_params = {
    # 'cm': 0.1,  # nF
    # 'v_reset': -70.,  # mV
    # 'v_rest': -65.,  # mV
    # 'v_thresh': -55.,  # mV
    # 'tau_m': 20.,  # ms
    # 'tau_refrac': 1.,  # ms
    # 'tau_syn_E': 5.0,  # ms
    # 'tau_syn_I': 5.0,  # ms
    
    'cm': 0.01,
    'v_reset': -70.,
    'v_rest': -20.,
    'v_thresh': -10,
    'e_rev_I': -100.,
    'e_rev_E': 60.,
    'tau_m': 20.,
    'tau_refrac': 0.1,
    'tau_syn_E': 5.,
    'tau_syn_I': 5.,

}

base_params['e_rev_I'] = -e_rev
base_params['e_rev_E'] = e_rev

duration = 10000.0
n_input = 30
n_mid = 5
n_per_mid = 50
w_in_to_mid = 1.0

hicanns_per_pop = [
[36,37,47,48,49,50],
[52,53,54,55,56,57],
[71,72,73,74,75,76],
[77,78,79,80,81],
[101,102,103,105,106],
]
in_pops = []
for i in range(n_input):
    p = sim.Population(1, sim.SpikeSourcePoisson, 
            {'rate': 200}, label="input %d"%i)
    # hicann = C.HICANNOnWafer(C.Enum(h))
    # marocco.manual_placement.on_hicann(p, hicann)
    p.record()
    in_pops.append(p)


mid_pops = []
for i in range(n_mid):
    p = sim.Population(n_per_mid, sim.IF_cond_exp, 
                       base_params, label="mid %d"%i)
    p.record()
    mid_pops.append(p)

inh_mid_pop = sim.Population(1, sim.IF_cond_exp, 
                            base_params, label="mid inh")


start_w = np.ones((n_input, n_mid * n_per_mid)) * np.nan
conn_lists = []
projs = []
for pre in range(n_input):
    p = []
    for post in range(n_mid):
        l = list_conn(1, n_per_mid, 0.2, w_in_to_mid, 1.0)
        for r, c, w, d in l:
            start_w[pre, post * n_per_mid + c] = w

        print(pre, ' to ', post, 'conns = ', len(l))
        conn_lists.append(l)
        prj = sim.Projection(in_pops[pre], mid_pops[post], 
                sim.FromListConnector(l), target='excitatory')
        p.append(prj)
    projs.append(p)


exc_wta_proj = []
inh_wta_proj = []
for idx in range(n_mid):
    proj = sim.Projection(mid_pops[idx], inh_mid_pop, 
                            sim.AllToAllConnector(), target='excitatory')
    exc_wta_proj.append(proj)

    proj = sim.Projection(inh_mid_pop, mid_pops[idx], 
                            sim.AllToAllConnector(), target='inhibitory')
    inh_wta_proj.append(proj)


# import sys
# sys.exit(0)

### ====================== PERFORM MAPPING =========================== ###
seed = 0
marocco.l1_routing.shuffle_switches_seed(seed)

marocco.skip_mapping = False
marocco.backend = PyMarocco.None

sim.reset()
sim.run(duration)

### ==================== DO A FIRST HARDWARE RUN ======================= ###

wafer = runtime.wafer()
hicanns_in_use = wafer.getAllocatedHicannCoordinates()

print("\n\n\n\n")
print(hicanns_in_use)

# for p in mid_pops:
    # if p.hicann is None:
        # continue

    # for hicann in p.hicann:
        # if hicann not in hicanns_in_use:
            # continue
        # _BSS_set_hicann_sthal_params(wafer, hicann, 1023)
        
for hicann in hicanns_in_use:
    print("\n\n\n\n")
    print(wafer, hicann)
    _BSS_set_hicann_sthal_params(wafer, hicann, 1023)


marocco.skip_mapping = True
marocco.backend = PyMarocco.Hardware
# Full configuration during first step
marocco.hicann_configurator = pysthal.ParallelHICANNv4Configurator()

set_digital_weights(start_w, projs, n_input, n_mid * n_per_mid, 
                    runtime, digital_w=15)

sim.run(duration)
in_spikes = [p.getSpikes() for p in in_pops]

spikes = [p.getSpikes() for p in mid_pops]

for s in in_spikes:
    print(len(s))

for popid, s in enumerate(spikes):
    print(popid, len(s))
    # print(s)

### ==================== DO A SECOND HARDWARE RUN ======================= ###
### ==================== zero-out some weights ======================= ###
sim.reset()
marocco.hicann_configurator = pysthal.NoResetNoFGConfigurator()
marocco.verification = PyMarocco.Skip
marocco.checkl1locking = PyMarocco.SkipCheck
# projections, total_in, total_out, bss_runtime, digital_w=15
out_weights = zero_digital_weights(projs, n_input, n_mid * n_per_mid, runtime)

sim.run(duration)

in_spikes = [p.getSpikes() for p in in_pops]
spikes = [p.getSpikes() for p in mid_pops]

sim.end()

for s in in_spikes:
    print(len(s))
    
# for s in spikes:
    # print(len(s))

n_spikes_per_out_neuron = [0 for _ in range(n_mid * n_per_mid)]
for popid, s in enumerate(spikes):
    print(popid, len(s))
    for nid, times in s:
        n_spikes_per_out_neuron[int(popid*n_per_mid + nid)] += 1

print(n_spikes_per_out_neuron)
print(len(n_spikes_per_out_neuron))

ow = out_weights.copy()
ow[np.isnan(ow)] = 0.0
sum_input_weights_per_out_neuron = []
for c in range(out_weights.shape[1]):
    sum_input_weights_per_out_neuron.append(ow[:, c].sum())

print(sum_input_weights_per_out_neuron)
print(len(sum_input_weights_per_out_neuron))

np.savez_compressed('data_output.npz', 
                    start_w=start_w, out_weights=out_weights, 
                    spikes=spikes, in_spikes=in_spikes,
                    n_spikes_per_out_neuron=n_spikes_per_out_neuron,
                    sum_input_weights_per_out_neuron=sum_input_weights_per_out_neuron,)
    

fig = plt.figure()
ax = plt.subplot(1, 1, 1)
plt.plot(sum_input_weights_per_out_neuron, n_spikes_per_out_neuron, '.')
ax.set_xlabel('Sum of input weights')
ax.set_ylabel('Total spikes')
plt.savefig('test_setting_digital_weights_to_zero_after_first_run.pdf')
plt.close(fig)

