#!/usr/bin/env python
# -*- coding: utf-8; -*-

import os
import numpy as np

from pyhalbe import HICANN
import pyhalbe.Coordinate as C
from pysthal.command_line_util import init_logger
import pysthal

import pyhmf as pynn
from pymarocco import PyMarocco, Defects
from pymarocco.runtime import Runtime
from pymarocco.coordinates import LogicalNeuron
from pymarocco.results import Marocco

init_logger("WARN", [
    # ("guidebook", "DEBUG"),
    # ("marocco", "DEBUG"),
    # ("Calibtic", "DEBUG"),
    ("guidebook", "INFO"),
    ("marocco", "INFO"),
    ("Calibtic", "INFO"),
    ("sthal", "INFO")
])

import pylogging
logger = pylogging.get("guidebook")


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

neuron_parameters = base_params

marocco = PyMarocco()
marocco.default_wafer = C.Wafer(int(os.environ.get("WAFER", 33)))
runtime = Runtime(marocco.default_wafer)
pynn.setup(marocco=marocco, marocco_runtime=runtime)

#  ——— set up network ——————————————————————————————————————————————————————————

pop = pynn.Population(1, pynn.IF_cond_exp, neuron_parameters)

pop.record()
pop.record_v()

hicann = C.HICANNOnWafer(C.Enum(297))
marocco.manual_placement.on_hicann(pop, hicann)

connector = pynn.AllToAllConnector(weights=1)

exc_spike_times = [
    250,
    # 500,
    # 520,
    # 540,
    # 1250,
]

inh_spike_times = [
    750,
    1000,
    1020,
    1040,
    1250,
]

duration = 500.0

stimulus_exc = pynn.Population(1, pynn.SpikeSourceArray, {
    'spike_times': exc_spike_times})
# stimulus_inh = pynn.Population(1, pynn.SpikeSourceArray, {
    # 'spike_times': inh_spike_times})

projections = [
    pynn.Projection(stimulus_exc, pop, connector, target='excitatory'),
    # pynn.Projection(stimulus_inh, pop, connector, target='inhibitory'),
]

#  ——— run mapping —————————————————————————————————————————————————————————————

marocco.skip_mapping = False
marocco.backend = PyMarocco.None

pynn.reset()
pynn.run(duration)

#  ——— change low-level parameters before configuring hardware —————————————————

def set_sthal_params(wafer, gmax, gmax_div):
    """
    synaptic strength:
    gmax: 0 - 1023, strongest: 1023
    gmax_div: 1 - 15, strongest: 1
    """

    # for all HICANNs in use
    for hicann in wafer.getAllocatedHicannCoordinates():

        fgs = wafer[hicann].floating_gates

        # set parameters influencing the synaptic strength
        for block in C.iter_all(C.FGBlockOnHICANN):
            fgs.setShared(block, HICANN.shared_parameter.V_gmax0, gmax)
            fgs.setShared(block, HICANN.shared_parameter.V_gmax1, gmax)
            fgs.setShared(block, HICANN.shared_parameter.V_gmax2, gmax)
            fgs.setShared(block, HICANN.shared_parameter.V_gmax3, gmax)

        for driver in C.iter_all(C.SynapseDriverOnHICANN):
            for row in C.iter_all(C.RowOnSynapseDriver):
                wafer[hicann].synapses[driver][row].set_gmax_div(
                    C.left, gmax_div)
                wafer[hicann].synapses[driver][row].set_gmax_div(
                    C.right, gmax_div)

        # don't change values below
        for ii in xrange(fgs.getNoProgrammingPasses()):
            cfg = fgs.getFGConfig(C.Enum(ii))
            cfg.fg_biasn = 0
            cfg.fg_bias = 0
            fgs.setFGConfig(C.Enum(ii), cfg)

        for block in C.iter_all(C.FGBlockOnHICANN):
            fgs.setShared(block, HICANN.shared_parameter.V_dllres, 275)
            fgs.setShared(block, HICANN.shared_parameter.V_ccas, 800)

# call at least once
gmax = 128
gmax_div = 1
set_sthal_params(runtime.wafer(), gmax=gmax, gmax_div=gmax_div)

#  ——— configure hardware ——————————————————————————————————————————————————————

marocco.skip_mapping = True
marocco.backend = PyMarocco.Hardware
# Full configuration during first step
marocco.hicann_configurator = pysthal.ParallelHICANNv4Configurator()

for digital_weight in range(1, 16):
    logger.info("running measurement with digital weight {}".format(digital_weight))
    for proj in projections:
        proj_items = runtime.results().synapse_routing.synapses().find(proj)
        print("\n---------------------------------------------------------")
        print("---------------------------------------------------------")
        print("---------------------------------------------------------")
        print(proj)
        print(proj_items)
        print(proj_items[0])
        synapse = proj_items[0].hardware_synapse()
        print(synapse)
        print(synapse.toHICANNOnWafer())
        proxy = runtime.wafer()[synapse.toHICANNOnWafer()].synapses[synapse]
        print(proxy)
        print(HICANN.SynapseWeight(digital_weight))
        proxy.weight = HICANN.SynapseWeight(digital_weight)
        print(proxy.weight)
        print("---------------------------------------------------------")
        print("---------------------------------------------------------")
        print("---------------------------------------------------------\n")
    pynn.run(duration)
    spikes = pop.getSpikes()
    volts = pop.get_v()
    
    if len(spikes):
        print("gmax {}\tgmaxd {}\tw{}\t{}".\
            format(gmax, gmax_div, digital_weight, len(spikes)))
    
    
    
    np.savetxt("membrane_gmax_{}_gmaxd_{}_w{}.txt".format(gmax, gmax_div, digital_weight), volts)
    np.savetxt("spikes_gmax_{}_gmaxd_{}_w{}.txt".format(gmax, gmax_div, digital_weight), spikes)
    pynn.reset()

    # only change digital parameters from now on
    marocco.hicann_configurator = pysthal.NoResetNoFGConfigurator()

# store the last result for visualization
runtime.results().\
    save("digital_weights_results_gmax_{}_gmaxd_{}.xml.gz".format(gmax, gmax_div), True)
