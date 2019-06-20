import numpy as np
import matplotlib.pyplot as plt
import operator
from collections import OrderedDict

def get_high_spiking(spikes, start_t, end_t, min_num_spikes):
    # print("\n\nIn get_high_spiking\n\n")
    # print(spikes)
    neurons = {}
    
    for nid, ts in enumerate(spikes):
        times = np.array(ts)
        whr = np.where(np.logical_and(start_t <= times, times < end_t))[0]
        
        if len(whr) > min_num_spikes:
            neurons[nid] = len(whr)
    
    # print(neurons)
    return sort_by_rate(neurons)


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
    up_w=0.1, down_w=0.1, max_w=np.inf, rand_add_prob=0.3, rand_del_prob=0.3,
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
                    if post_ts.size > 0 and max_pre_t < max_post_t: # and post spiked as well after
                        ws[pre, post] += up_w # increase synapse
                    elif np.random.uniform(0., 1.) <= rand_add_prob: # post didn't spike but randomly increase synapses
                        ws[pre, post] += up_w
                    else: # and if post didn't spike, not sensitive to pattern
                        ws[pre, post] -= down_w # decrease synapse

                elif post_ts.size > 0: # pre didn't spike but post did
                        ws[pre, post] -= down_w # pre is not part of the pattern, decrease synapse
                elif np.random.uniform(0., 1.) <= rand_del_prob: # we have no pairs, randomly reduce synapse
                        ws[pre, post] -= down_w


    #keep noisy neurons at bay ... hopefully
    print(pre_blacklist)
    for pre in pre_blacklist: 
        pre = int(pre)
        ws[pre, :] = 0

    print(post_blacklist)
    for post in post_blacklist:
        post = int(post)
        ws[:, post] = 0

    ws[:] = np.clip(0.0, max_w, ws) ### keep weights positive

    return ws
