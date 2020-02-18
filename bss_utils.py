import numpy as np
import matplotlib.pyplot as plt
import operator
from collections import OrderedDict
# from spikevo.partitioning import SplitProjection
from pyhalbe import HICANN
import pyhalbe.Coordinate as C
from pymarocco import PyMarocco
from pymarocco.runtime import Runtime
from pymarocco.coordinates import LogicalNeuron
from pymarocco.results import Marocco
from pymarocco import Defects
import pysthal
import copy

SYNAPSE_DECODER_DISABLED_SYNAPSE = HICANN.SynapseDecoder(1)


def secs_to_hms(seconds):
    mins_to_run = seconds // 60
    seconds -= mins_to_run * 60
    hours_to_run = mins_to_run // 60
    mins_to_run -= hours_to_run * 60
    seconds, mins_to_run, hours_to_run = int(seconds), int(mins_to_run), int(hours_to_run)

    return '{:02d}h: {:02d}m: {:02d}s'.format(hours_to_run, mins_to_run, seconds)


def reduce_influence(neuron_ids, weights, pre_or_post, n_to_delete):
    # print("in reduce_influence")
    # print(neuron_ids)
    if pre_or_post.lower() == 'post':
        for _id in neuron_ids:
            on_w = np.where(np.logical_and(~ np.isnan(weights[:, _id]),
                                        weights[:, _id] > 0))[0]
            ntd =  n_to_delete if len(on_w) >= n_to_delete else len(on_w)
            # print(ntd)
            del_ids = np.random.choice(on_w, size=ntd, replace=False)
            weights[del_ids, _id] = np.nan
    else:
        for _id in neuron_ids:
            on_w = np.where(np.logical_and(~ np.isnan(weights[_id, :]),
                                        weights[_id, :] > 0))[0]
            ntd =  n_to_delete if len(on_w) >= n_to_delete else len(on_w)
            del_ids = np.random.choice(on_w, size=ntd, replace=False)
            weights[_id, del_ids] = np.nan
                
    
    return weights

def _BSS_set_hicann_sthal_params(wafer, hicann, gmax, gmax_div=1):
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



def _BSS_set_sthal_params(BSS_runtime, gmax=1023, gmax_div=1):
    """
    synaptic strength:
    gmax: 0 - 1023, strongest: 1023
    gmax_div: 1 - 15, strongest: 1
    """
    wafer = BSS_runtime.wafer()
    # for all HICANNs in use
    for hicann in wafer.getAllocatedHicannCoordinates():
        _BSS_set_hicann_sthal_params(wafer, hicann, gmax, gmax_div)



def set_digital_weights(weights, projection, bss_runtime, digital_w=15):
    total_in, total_out = weights.shape

    original_decoders = {}

    rtime_res = bss_runtime.results()
    synapses = rtime_res.synapse_routing.synapses()
    proj_items = synapses.find(projection)

    for _item in proj_items:
        syn = _item.hardware_synapse()
        pre = _item.source_neuron().neuron_index()
        post = _item.target_neuron().neuron_index()
        w = weights[pre, post]

        proxy = bss_runtime.wafer()[syn.toHICANNOnWafer()].synapses[syn]
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


def list_to_matrix(pre_pop, post_pop, _conn_list):
    mtx = np.ones((pre_pop.size, post_pop.size)) * np.nan
    for pre, post, w, d in _conn_list:
        mtx[pre, post] = w
    return mtx


def get_hicanns(center_hicann, n_kenyon, seed=1, max_dist=3, n_per_pop=4, manual=False):
    if manual:
        f = open("black_list_stats.txt", "a+")
        f.write(u"%s, -*-\n"%seed)
        f.close()

        np.random.seed(seed)
        ID, ROW, COL = range(3)
        w = WAL()
        hood = w.get_neighbours(center_hicann, max_dist=max_dist)
        ids = []
        for r in hood:
            for c in hood[r]:
                ids.append(hood[r][c][ID])

        pprint(hood)
        print(ids)
        
        pops = ['antenna', 'decision', 
                'feedback', 'exciter',
                # 'tick', 'exciter_src',
                # 'kenyon',
                ]
        ### ideal config is in a 3x3 grid
        places = {}
        blacklist = []
        # ## blacklist = [73, 76, 99, 17, 18, 19, 20, 21, 37, 47, 167, 56, 6, 5, 7, 80, 100, 14]
        used = [] + blacklist
                
        k_places = []
        for i in range(n_kenyon):
            avail = np.setdiff1d(ids, used)
            np.random.choice(avail, size=n_kenyon, replace=False)
            hicann_id = np.random.choice(avail, size=n_per_pop)

            hicann = [C.HICANNOnWafer(C.Enum(i)) for i in hicann_id]
            for i in hicann_id:
                used.append(i)
            k_places.append(hicann)
            
        places['kenyon'] = k_places

        for p in pops:
            avail = np.setdiff1d(ids, used)
            hicann_id = np.random.choice(avail, size=n_per_pop)
            hicann = [C.HICANNOnWafer(C.Enum(i)) for i in hicann_id]
            places[p] = hicann
            for i in hicann_id:
                used.append(i)


        
        for k in sorted(places):
            for p in places[k]:
                try:
                    sys.stdout.write("{},".format(int(p.id())))
                except:
                    for q in p:
                        sys.stdout.write("{},".format(int(q.id())))
        print()
        print(places)
    else:
        places = {
            'antenna': None,
            'kenyon': [None] * n_kenyon,
            'decision': None,
            'tick': None,
            'feedback': None,
            'exciter src': None,
            'exciter': None,
        }
    
    return places

def all_to_all(pre_pop, post_pop, weights=1.0, delays=1.0):
    return [[pre, post, weights, delays] 
            for pre in range(pre_pop.size) for post in range(post_pop.size)]

def input_connection_list(input_size, kenyon_size, prob_conn, weight, seed=111):
    matrix = np.ones((input_size * kenyon_size, 4))
    matrix[:, 0] = np.repeat(np.arange(input_size), kenyon_size)
    matrix[:, 1] = np.tile(np.arange(kenyon_size), input_size)

    np.random.seed(seed)

    matrix[:, 2] = 0

    dice = np.random.uniform(0., 1., size=(input_size * kenyon_size))
    active = np.where(dice <= prob_conn)[0]
    
    conn_lists = [[] for _ in range(input_size)]
    for conn_id in active:
        pre_id = conn_id // kenyon_size
        post_id = conn_id % kenyon_size
        conn_lists[pre_id].append( (0, post_id, weight, 1.0) )
        
    np.random.seed()

    return conn_lists
    

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

STRINGABLE = [
    'nAL', 'nKC', 'nDN', 'probAL', 
    'probNoiseSamplesAL', 'nPatternsAL'
]
def args_to_str(arguments, stringable=STRINGABLE):

    d = vars(arguments)
    arglist = []
    for arg in d:
        v = str(d[arg])
        if arg not in stringable:
            continue
        v = v.replace('.', 'p')
        arglist.append('{}_{}'.format(arg, v))

    return '__'.join(arglist)


def get_high_spiking(spikes, start_t, end_t, min_num_spikes):
    # print("\n\nIn get_high_spiking\n\n")
    # print(spikes)
    neurons = {}
    
    for nid, t in spikes:
        if start_t <= t and t < end_t:
            l = neurons.get(nid, [])
            l.append( t )
            neurons[nid] = l

    hi = [int(nid) for nid in neurons if len(neurons[nid]) >= min_num_spikes]

    # print(neurons)
    # return sort_by_rate(neurons)
    return hi


def sort_by_rate(hi_spiking):
    # print(hi_spiking.items())
    return OrderedDict(sorted(\
            hi_spiking.items(), key=operator.itemgetter(1), reverse=True))


def bin_spikes_per_sample(start_t, end_t, sample_dt, spikes):
    st = 0
    et = 0
    total_t = int(end_t - start_t)
    n_bins = int(total_t//sample_dt + int(total_t%sample_dt > 0))
    binned = [[np.empty(0) for _ in spikes] for _ in range(n_bins)]
    for st in np.arange(start_t, end_t, sample_dt):
        et = st + sample_dt
        bin_id = int((st - start_t) // sample_dt)
        for neuron_id, ts in enumerate(spikes):
            _all = np.array(ts)
            _times = _all[np.where(np.logical_and(st <= _all, _all < et))]
            if _times.size > 0:
                binned[bin_id][neuron_id] = np.append(binned[bin_id][neuron_id], _times)
    return binned


DEFAULT_SP_CONFIG = dict(
    up_w=0.1, down_w=0.1, max_w=np.inf, rand_add_prob=0.1, rand_del_prob=0.1,
)
def structural_plasticity(pre_spikes_binned, post_spikes_binned, weights, 
                          pre_blacklist, post_blacklist, config=DEFAULT_SP_CONFIG):
    up_w, down_w = config['up_w'], config['down_w']
    max_w = config['max_w']
    rand_add_prob = config['rand_add_prob']
    rand_del_prob = config['rand_del_prob']

    ws = weights.copy()
    num_bins = len(pre_spikes_binned)
    for bin_id in range(num_bins):
        pre_bin = pre_spikes_binned[bin_id]
        post_bin = post_spikes_binned[bin_id]
        for pre, pre_ts in enumerate(pre_bin):
            max_pre_t = np.inf if pre_ts.size == 0 else pre_ts.max()
            for post, post_ts in enumerate(post_bin):
                max_post_t = -np.inf if post_ts.size == 0 else post_ts.max()
                # should we randomize these actions?
                if pre_ts.size > 0: # pre spiked for this pattern
                    if post_ts.size > 0:# and max_pre_t < max_post_t: # and post spiked as well after
                        ws[pre, post] += up_w # increase synapse
                    elif np.random.uniform(0., 1.) <= rand_add_prob: # post didn't spike but randomly increase synapses
                        ws[pre, post] += up_w
                    # else: # and if post didn't spike, not sensitive to pattern
                        # ws[pre, post] -= down_w # decrease synapse

                # elif post_ts.size > 0: # pre didn't spike but post did
                        # ws[pre, post] -= down_w # pre is not part of the pattern, decrease synapse
                # elif np.random.uniform(0., 1.) <= rand_del_prob: # we have no pairs, randomly reduce synapse
                        # ws[pre, post] = 0


    # #keep noisy neurons at bay ... hopefully
    # for pre in pre_blacklist: 
    #     pre = int(pre)
    #     ws[pre, :] = 0

    # for post in post_blacklist:
        # post = int(post)
        # ws[:, post] = 0

    ws[:] = np.clip(ws, 0.0, max_w) ### keep weights positive

    return ws

