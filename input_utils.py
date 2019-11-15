from __future__ import (print_function,
                        unicode_literals,
                        division)
from future.builtins import str, open, range, dict
import numpy as np
import sys
import os

### from NE15/poisson_tools.py
def nextTime(rateParameter):
    '''Helper function to Poisson generator
       :param rateParameter: The rate at which a neuron will fire (Hz)

       :returns: Time at which the neuron should spike next (seconds)
    '''
    return -np.log(1.0 - np.random.uniform(0.0, 1.0)) / rateParameter
    # random.expovariate(rateParameter)


def poisson_generator(rate, t_start, t_stop):
    '''Poisson train generator
       :param rate: The rate at which a neuron will fire (Hz)
       :param t_start: When should the neuron start to fire (milliseconds)
       :param t_stop: When should the neuron stop firing (milliseconds)

       :returns: Poisson train firing at rate, from t_start to t_stop (milliseconds)
    '''
    poisson_train = []
    if rate > 0:
        next_isi = nextTime(rate) * 1000.
        last_time = next_isi + t_start
        while last_time < t_stop:
            poisson_train.append(last_time)
            next_isi = nextTime(rate) * 1000.
            last_time += next_isi
    return poisson_train


def generate_input_vectors(num_vectors, dimension, on_probability, seed=1):

    n_active = int(on_probability*dimension)
    fname = 'vectors_{}_{}_{}_{}.npz'.format(num_vectors, dimension, n_active, seed)
    if os.path.isfile(fname):
        f = np.load(fname, allow_pickle=True)
        return f['vectors']

    np.random.seed(seed)
    # vecs = (np.random.uniform(0., 1., (num_vectors, dimension)) <= on_probability).astype('int')
    vecs = np.zeros((num_vectors, dimension))
    for i in range(num_vectors):
        indices = np.random.choice(np.arange(dimension, dtype='int'), size=n_active, replace=False)
        vecs[i, indices] = 1.0
    np.random.seed()

    np.savez_compressed(fname, vectors=vecs)

    return vecs

def generate_spike_times_poisson(input_vectors, num_samples, sample_dt, start_dt, high_dt,
                                 high_freq, low_freq, seed=1, randomize_samples=True, regenerate=False):

    fname = 'poission_spike_times_n{}_v{}_dim{}_dt{}_hi{}_seed{}.npz'.format(
        num_samples, input_vectors.shape[0], input_vectors.shape[1], sample_dt, high_freq, seed)

    if not regenerate and os.path.isfile(fname):
        f = np.load(fname, allow_pickle=True)
        return f['indices'], f['spike_times'].tolist()

    np.random.seed(seed)
    spike_times = [[] for _ in range(input_vectors.shape[1])]
    total_samples = num_samples * input_vectors.shape[0]
    if randomize_samples:
        indices = np.arange(total_samples)
        np.random.shuffle(indices)
    else:
        indices = np.arange(total_samples)

    t = 0
    for i, s_idx in enumerate(indices):
        sys.stdout.write('\r\t\t%6.2f%%' % (100 * ((i + 1.0) / total_samples)))
        sys.stdout.flush()

        pat_idx = s_idx // num_samples
        pat = input_vectors[pat_idx]
        for n_idx in range(pat.size):
            if pat[n_idx] == 0:
                spike_times[n_idx] += poisson_generator(low_freq, t, t + sample_dt)
            else:
                start_t = t + start_dt
                end_t = start_t + high_dt
                spike_times[n_idx] += poisson_generator(high_freq, start_t, end_t)

        t += sample_dt

    for n_idx in range(pat.size):
        spike_times[n_idx][:] = sorted(spike_times[n_idx])

    np.savez_compressed(fname, spike_times=spike_times, indices=indices)

    np.random.seed()

    sys.stdout.write('\n')
    sys.stdout.flush()

    return indices, spike_times


def generate_samples(input_vectors, num_samples, n_test, prob_noise, seed=1, method='exact', regenerate=False):
    """method='all' means randomly choose indices where we flip 1s and 0s with probability = prob_noise"""
    np.random.seed(seed)

    fname = 'samples_{}_{}_{}_{}.npz'.format(
        input_vectors.shape[0], input_vectors.shape[1], num_samples, seed)

    if os.path.isfile(fname) and not regenerate:
        f = np.load(fname, allow_pickle=True)
        return f['samples']

    samples = None

    for i in range(input_vectors.shape[0]):
        sys.stdout.write('\r\t\t%6.2f%%' % (100 * ((i + 1.0) / input_vectors.shape[0])))
        sys.stdout.flush()

        samp = np.tile(input_vectors[i, :], (num_samples, 1)).astype('int')
        if method == 'random':
            base_flips = int(np.round(np.mean(input_vectors.sum(axis=1)) * prob_noise))
            for j in range(num_samples-n_test):
                # flip zeros to ones
                n_flips = base_flips #+ np.random.randint(-1, 2)
                indices = np.random.choice(np.where(samp[j] == 0)[0], size=n_flips, replace=False)
                samp[j, indices] = 1

                #flip ones to zeros
                n_flips = base_flips + np.random.randint(0, 3)
                indices = np.random.choice(np.where(samp[j] == 1)[0], size=n_flips, replace=False)
                samp[j, indices] = 0

        elif method == 'exact':
            n_flips = int(np.round(np.mean(input_vectors.sum(axis=1)) * prob_noise))
            for j in range(num_samples-n_test):
                #flip ones to zeros
                indices = np.random.choice(np.where(samp[j] == 1)[0], size=n_flips, replace=False)
                samp[j, indices] = 0

                # flip zeros to ones
                indices = np.random.choice(np.where(samp[j] == 0)[0], size=n_flips, replace=False)
                samp[j, indices] = 1
        else:
            dice = np.random.uniform(0., 1., samp.shape)
            whr = np.where(dice[:-n_test, :] < prob_noise)
            samp[whr] = 1 - samp[whr]

        if samples is None:
            samples = samp
        else:
            samples = np.append(samples, samp, axis=0)

    np.random.seed()

    np.savez_compressed(fname, samples=samples)

    sys.stdout.write('\nSamples shape = {}\n'.format(samples.shape))
    sys.stdout.flush()

    return samples

def samples_to_spike_times(samples, n_samples, n_test, sample_dt, start_dt, max_rand_dt, sim_timestep, seed=1,
    randomize_samples=False, regenerate=False):

    fname = 'spike_times_{}_{}_{}_{}_{}.npz'.format(
        samples.shape[0], samples.shape[1], sample_dt, start_dt, seed)

    if not regenerate and os.path.isfile(fname):
        f = np.load(fname, allow_pickle=True)
        return f['indices'], f['spike_times'].tolist()



    spike_times = [[] for _ in range(samples.shape[-1])]
    total_samples = samples.shape[0]
    n_train = n_samples - n_test
    orig_indices = np.tile(np.arange(n_samples), total_samples // n_samples)
    train_indices = np.where(orig_indices < n_train)[0]
    test_indices = np.where(orig_indices >= n_train)[0]
    
    if randomize_samples:
        np.random.seed()
        np.random.shuffle(train_indices)
        np.random.shuffle(test_indices)
        np.random.seed()
        np.random.shuffle(train_indices)
        np.random.shuffle(test_indices)
        # np.random.shuffle(indices)

    np.random.seed(seed)
    t = 0
    total = float(len(orig_indices))
    for i, idx in enumerate(train_indices):
        sys.stdout.write('\r\t\t%6.2f%%'%(100*((i+1.0)/total)))
        sys.stdout.flush()

        samp = samples[idx]
        active = np.where(samp == 1.)[0]
        # max_start_dt = (sample_dt - start_dt)
        rand_start_dt = 0#np.random.randint(-start_dt, start_dt)
        rand_dt = np.random.randint(-max_rand_dt, max_rand_dt+1, size=active.size).astype('float') \
                    if max_rand_dt > 0 else np.zeros(active.shape)
        rand_dt *= sim_timestep
        ts = t + rand_start_dt + start_dt + rand_dt
        for time_id, neuron_id in enumerate(active):
            if ts[time_id] not in spike_times[neuron_id]:
                spike_times[neuron_id].append(ts[time_id])

        t += sample_dt


    for j, idx in enumerate(test_indices):
        sys.stdout.write('\r\t\t%6.2f%%'%(100*((i+j+1.0)/total)))
        sys.stdout.flush()

        samp = samples[idx]
        active = np.where(samp == 1.)[0]
        # max_start_dt = (sample_dt - start_dt)
        rand_start_dt = 0#np.random.randint(-start_dt, start_dt)
        rand_dt = np.random.randint(-max_rand_dt, max_rand_dt+1, size=active.size).astype('float') \
                    if max_rand_dt > 0 else np.zeros(active.shape)
        rand_dt *= sim_timestep
        ts = t + rand_start_dt + start_dt + rand_dt
        for time_id, neuron_id in enumerate(active):
            if ts[time_id] not in spike_times[neuron_id]:
                spike_times[neuron_id].append(ts[time_id])

        t += sample_dt


    
    np.random.seed()

    sys.stdout.write('\n')
    sys.stdout.flush()
    
    indices = np.append(train_indices, test_indices)
    
    np.savez_compressed(fname, spike_times=spike_times, indices=indices)

    return indices, spike_times


def generate_tick_spikes(samples, sample_dt, start_dt, num_test_samples, delay):
    n_samples = (samples.shape[0] - num_test_samples)
    t = start_dt + delay
    ticks = [[]]
    for _ in range(n_samples):
        ticks[0].append(t)
        t += sample_dt

    return ticks
