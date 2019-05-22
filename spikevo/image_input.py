from __future__ import (print_function,
                        # unicode_literals,
                        division)
from builtins import str, open, range, dict

import numpy as np
from . import *

class SpikeImage(object):
    
    def __init__(self, width, height, encoding=ON_OFF, 
        encoding_params={'rate': 100, 'threshold': 12}):
        if encoding not in IMAGE_ENCODINGS:
            raise Exception("Image encoding not supported ({})".format(encoding))

        self._width = width
        self._height = height
        self._size = width * height
        self._encoding = encoding
        self._enc_params = encoding_params
        
        
    def _encode_on_off(self, source):
        flat = source.flatten()
        rate = self._enc_params['rate']
        threshold = self._enc_params['threshold']
        off_pixels = np.where(flat < threshold)[0]
        on_pixels = np.where(flat > threshold)[0]
        
        return {ON: [rate*(flat[i] - threshold) if i in on_pixels else 0 \
                                            for i in range(self._size)],
                OFF: [rate*(threshold - flat[i]) if i in off_pixels else 0 \
                                            for i in range(self._size)]}


    def _encode_rate(self, source):
        flat = source.flatten()
        rate = self._enc_params['rate']
        return {OFF: [rate*flat[i] for i in range(self._size)]}


    def encode(self, source):
        if self._encoding == RATE:
            return self._encode_rate(source)
        elif self._encoding == ON_OFF:
            return self._encode_on_off(source)


    def rate_to_poisson(self, spike_rates, start_time, end_time):
        def nextTime(rateParameter):
            return -np.log(1.0 - np.random.random()) / rateParameter

        def poisson_generator(rate, t_start, t_stop):
            poisson_train = []
            if rate > 0:
                next_isi = nextTime(rate)*1000.
                last_time = next_isi + t_start
                while last_time  < t_stop:
                    poisson_train.append(last_time)
                    next_isi = nextTime(rate)*1000.
                    last_time += next_isi
            return poisson_train
        
        return [poisson_generator(rate, start_time, end_time) \
                                        for rate in spike_rates]


    def spikes_to_image(self, spike_trains, tstart=0, tend=np.inf, 
        spike_val=1):
        """spike_trains is a dictionary containing spikes from each channel
            (2 for ON_OFF encoding, 1 for RATE encoding)
            
            Spike times have to be in a format compatible with the PyNN 0.8+
            [[n0t0, n0t1], [n1t0, n1t1, n1t2] ... [nNt0]]
        """
        channels = 3 if self._encoding == ON_OFF else 1
        img = np.zeros((self._height, self._width, channels))
        for ch in spike_trains:
            color = CHAN2COLOR[ch]
            for nid, spike_times in enumerate(spike_trains[ch]):
                if len(spike_times) == 0:
                    continue
                row, col = nid//self._width, nid%self._width
                # print(CHAN2TXT[ch], row, col, spike_times)
                print(CHAN2TXT[ch], row, col)
                for t in spike_times:
                    img[row, col, color] += spike_val
        
        return img

    def create_pops(self, pynnx, neuron_parameters={ON:{}, OFF:{}},
        generation_type='array'):
        if generation_type == 'array':
            return self.create_pops_array(pynnx, neuron_parameters=neuron_parameters)
        elif generation_type == 'poisson':
            return self.create_pops_poisson(pynnx, neuron_parameters=neuron_parameters)

    def create_pops_poisson(self, pynnx, neuron_parameters={ON:{}, OFF:{}}):
        if self._encoding == RATE:
            """OFF means first and only channel for RATE encoding,
                this gets translated into a RED == 0 index in the image
            """
            if pynnx._sim_name == NEST:
                return {
                    OFF: NestImagePopulation(pynnx,
                        self._size, pynnx.sim.SpikeSourcePoisson, 
                        neuron_parameters, label='Rate encoded image')
                }
            else:
                return {
                    OFF: pynnx.Pop(self._size, pynnx.sim.SpikeSourcePoisson, 
                        neuron_parameters, label='Rate encoded image')
                }
        elif self._encoding == ON_OFF:
            if pynnx._sim_name == NEST:
                return {
                    OFF: NestImagePopulation(pynnx,
                        self._size, pynnx.sim.SpikeSourcePoisson, 
                        neuron_parameters[OFF], label='OFF - rate encoded image'),
                    ON: NestImagePopulation(pynnx,
                        self._size, pynnx.sim.SpikeSourcePoisson, 
                        neuron_parameters[ON], label='ON - rate encoded image')
                }
            else:
                return {
                    OFF: pynnx.Pop(self._size, pynnx.sim.SpikeSourcePoisson, 
                        neuron_parameters[OFF], label='OFF - rate encoded image'),
                    ON: pynnx.Pop(self._size, pynnx.sim.SpikeSourcePoisson, 
                        neuron_parameters[ON], label='ON - rate encoded image')
                }

    def create_pops_array(self, pynnx, neuron_parameters={ON:{}, OFF:{}}):
        if self._encoding == RATE:
            """OFF means first and only channel for RATE encoding,
                this gets translated into a RED == 0 index in the image
            """
            if pynnx._sim_name == NEST:
                return {
                    OFF: NestImagePopulation(pynnx, self._size, 
                        pynnx.sim.SpikeSourceArray, 
                        neuron_parameters, label='Rate encoded image')
                }
            else:
                return {
                    OFF: pynnx.Pop(self._size, pynnx.sim.SpikeSourceArray, 
                        neuron_parameters, label='Rate encoded image')
                }
        elif self._encoding == ON_OFF:
            if pynnx._sim_name == NEST:
                return {
                    OFF: NestImagePopulation(pynnx,
                        self._size, pynnx.sim.SpikeSourceArray, 
                        neuron_parameters[OFF], label='OFF - rate encoded image'),
                    ON: NestImagePopulation(pynnx,
                        self._size, pynnx.sim.SpikeSourceArray, 
                        neuron_parameters[ON], label='ON - rate encoded image')
                }
            else:
                return {
                    OFF: pynnx.Pop(self._size, pynnx.sim.SpikeSourceArray, 
                        neuron_parameters[OFF], label='OFF - rate encoded image'),
                    ON: pynnx.Pop(self._size, pynnx.sim.SpikeSourceArray, 
                        neuron_parameters[ON], label='ON - rate encoded image')
                }


class NestImagePopulation(object):
    def __init__(self, pynnal, size, neuron_class, neuron_params, label=''):
        if NEST not in pynnal._sim_name:
            raise Exception(
                "Nest Image cannot be instantiated with the current backend ({})".\
                    format(pynnal._sim_name))
        self.size = size
        self._pynnal = pynnal
        self._neuron_class = neuron_class
        self._neuron_params = neuron_params
        self._label = label
        
        self.setup_cam_pop(size, neuron_class, neuron_params, label=label)
        
    def record(self, what):
        self.out.record(what)
    
    def getSpikes(self, **kwargs):
        return self.out.getSpikes(**kwargs)
        
    def set(self, attr, val):
        for i in len(val):
            self._dummy_pops[i].set(attr, val)
    
    def setup_cam_pop(self, pop_size, neuron_class, neuron_parameters, label=''):
        
        param_name = 'spike_times' if 'Array' in '%s'%neuron_class else 'rate'
        pynnx = self._pynnal
        sim = pynnx.sim
        out_cell = sim.IF_curr_exp
        out_params = { 
            'cm': 0.25, 'i_offset': 0.0, 'tau_m': 10.0, 'tau_refrac': 3.0,
            'tau_syn_E': 1., 'tau_syn_I': 2., 'v_reset': -80.0,
            'v_rest': -65.0, 'v_thresh': -55.4
        }
        w2s = 4.5
        dmy_pops = []
        dmy_prjs = []
        cam_pop = sim.Population(pop_size, out_cell, out_params, label=label)

        for i in range(pop_size):
            dmy_pops.append(
                sim.Population(1, neuron_class, 
                    {param_name: neuron_parameters[param_name][i]},
                    label='{} pixel ({})'.format(label, i)))
            conn = [(0, i, w2s, 1)]
            dmy_prjs.append(
                sim.Projection(dmy_pops[i], cam_pop,
                    sim.FromListConnector(conn),
                    target='excitatory', label='dmy to cam {}'.format(i)))


        self.out = cam_pop
        self._dummy_pops = dmy_pops
        self._dummy_projs = dmy_prjs
