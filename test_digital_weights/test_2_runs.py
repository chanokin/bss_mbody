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
import numpy as np
import copy

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
    return [(pre, post, w, d) for pre in range(n_pre) for n_post in range(n_post) \
                if np.random.uniform() <= prob]

def zero_digital_weights(projections, total_in, total_out, bss_runtime, digital_w=2):
    n_per_out = total_out // len(projections[0])
    in_mtx = np.zeros((total_in, total_out))
    for i, prjs in enumerate(projections):
        for j, p in enumerate(prjs):
            c0 = j * n_per_out
            c1 = c0 + n_per_out
            in_mtx[i, c0:c1] = p.getWeights(format='array')
    out_mtx = in_mtx.copy()

    whr = np.where(~np.isnan(out_mtx))
    zeros = np.where(np.random.uniform(size=whr[0].shape) < 0.5)
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

                if np.isnan(w) or w < 0.0:
                    proxy.weight = HICANN.SynapseWeight(0)
                    proxy.decoder = SYNAPSE_DECODER_DISABLED_SYNAPSE
                else:
                    proxy.weight = HICANN.SynapseWeight(digital_w)
                    proxy.decoder = original_decoders[syn]



############################################################################
############################################################################
############################################################################

wafer = 33
marocco = PyMarocco()
marocco.backend = PyMarocco.Hardware

marocco.default_wafer = C.Wafer(wafer)
runtime = Runtime(marocco.default_wafer)
calib_path = "/wang/data/calibration/brainscales/WIP-2018-09-18"

marocco.calib_path = calib_path
marocco.defects.path = marocco.calib_path
marocco.verification = PyMarocco.Skip
marocco.checkl1locking = PyMarocco.SkipCheck
marocco.continue_despite_synapse_loss = True

SYNAPSE_DECODER_DISABLED_SYNAPSE = HICANN.SynapseDecoder(1)

### ====================== NETWORK CONSTRUCTION =========================== ###
sim.setup(timestep=1.0, min_delay=1.0, marocco=marocco, marocco_runtime=runtime)

duration = 10000.0
n_input = 50
n_mid = 5
n_per_mid = 50
w_in_to_mid = 1.0

in_pops = []
for i in range(n_input):
    p = sim.Population(1, sim.SpikeSourcePoisson, {'rate': 50}, label="input %d"%i)
    in_pops.append(p)


mid_pops = []
for i in range(n_mid):
    p = sim.Population(n_per_mid, sim.IF_cond_exp, {}, label="mid %d"%i)
    p.record('spikes')
    mid_pops.append(p)

conn_lists = []
projs = []
for pre in range(n_input):
    p = []
    for post in range(n_mid):
        l = list_conn(1, n_per_mid, w_in_to_mid, 1.0)
        conn_lists.append(l)
        p.append(sim.Projection(pre, post, sim.FromListConnector(l)))
    projs.append(p)


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

for p in mid_pops:
    if p.hicann is None:
        continue

    for hicann in p.hicann:
        if hicann not in hicanns_in_use:
            continue

        _BSS_set_hicann_sthal_params(wafer, hicann, 1023)

marocco.skip_mapping = True
marocco.backend = PyMarocco.Hardware
# Full configuration during first step
marocco.hicann_configurator = pysthal.ParallelHICANNv4Configurator()

sim.run(duration)


### ==================== DO A SECOND HARDWARE RUN ======================= ###
### ==================== zero-out some weights ======================= ###
sim.reset()
marocco.hicann_configurator = pysthal.NoResetNoFGConfigurator()
marocco.verification = PyMarocco.Skip
marocco.checkl1locking = PyMarocco.SkipCheck

out_weights = zero_digital_weights(projs)
sim.run(duration)

spikes = [p.getSpikes() for p in mid_pops]

sim.end()

