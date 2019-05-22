from __future__ import (print_function,
                        # unicode_literals,
                        division)
from builtins import open, range, dict
 
import Pyro4
from spikevo.pynn_transforms import PyNNAL
from spikevo import *
import sys
import traceback
from multiprocessing import Process, Queue

@Pyro4.expose
# @Pyro4.behavior(instance_mode="percall") #single instance per call
@Pyro4.behavior(instance_mode="session") #single instance per proxy connection
class NeuralNetworkServer(object):
    def __init__(self):
        self._processes = {}
        self._queues = {}

    def full_run(self, run_spec, simulator_name, description, multiprocessor=False, label=None,
                timestep=1.0, min_delay=1.0, per_sim_params={}, recordings=None, weights=None):
        def fr(queue, run_spec, simulator_name, description, multiprocessor, label,
               timestep, min_delay, per_sim_params, recordings, weights):
            # sys.stderr.write("\n\nIn full_run.fr\n")
            # sys.stderr.flush()

            decoder = NeuralNetworkDecoder()
            decoder.set_net(simulator_name, description, multiprocessor, label, timestep,
                           min_delay, per_sim_params)
            data = {}
            for k in run_spec:
                if 'settings' in run_spec[k]:
                    pop = run_spec[k]['settings']['pop']
                    for var in run_spec[k]['settings']['values']:
                        value = run_spec[k]['settings']['values'][var]
                        decoder.set_var(pop, var, value)

                decoder.run(run_spec[k]['time'], recordings, label)
                recs = decoder.get_records(label)
                end_weights = {}
                if weights is not None:
                    for source, target in weights:
                        w_dict = end_weights.get(source, {})
                        w_dict[target] = (decoder.get_weights(decoder.projections[source][target])).tolist()
                        end_weights[source] = w_dict

                data[k] = {'recordings': recs, 'weights': end_weights}

            queue.put()
            decoder.end(label)

        if np.isscalar(run_spec):
            run_spec = {0: {'time': run_spec}}

        queue = Queue()
        args = (queue, run_spec, simulator_name, description, multiprocessor,
                label, timestep, min_delay, per_sim_params, recordings, weights)
        proc = Process(target=fr, args=args)
        proc.start()
        data = queue.get()
        self._processes[label] = proc

        return data

    def __del__(self):
        for p in self._processes:
            self._processes[p].join()

class NeuralNetworkDecoder(object):
    def __init__(self):
        self.initialized = False
        self.record_ids = {}
        self.sim = {}
        self.pynnx = {}
        self.labels = []
        self.populations = {}
        self.projections = {}
        self.processes = {}

    def check_multiproc(self, label):

        # sys.stderr.write("\n\nIn NNServer.check_multiproc\n")
        # sys.stderr.write("self.multiprocessor = %s\n"%self.multiprocessor)
        # sys.stderr.write("label = %s\n\n"%label)
        # sys.stderr.flush()
        if self.multiprocessor and label is None:
            traceback.print_stack()
            raise Exception('When multiprocessor flag is true a label is needed for '
                            'each processes; no label was provided')
        elif not self.multiprocessor:
            return 0
        else:
            return label

    def set_net(self, simulator_name, description, multiprocessor=False, label=None,
                timestep=1.0, min_delay=1.0, per_sim_params={}):
        """Wrapper to execute a PyNN script given a simulator and the network description
        simulator_name: The name of the backend, it will be used to import the correct
            libraries.
        description: A dictionary containing population and projection descriptors.
            Population descriptors must include standard PyNN requirements such as 
            population size, neuron type, parameters (including spike times, rates).
            Projection descriptors must include standard PyNN requirements such as
            source and target populations, connector type and its parameters, plasticity.
        timestep: Simulation timestep (ms)
        min_delay: Minimum delay in synaptic connections between populations (ms)
        per_sim_params: Extra parameters needed for specific simulator (e.g. 
            wafer id for BrainScaleS, max_neurons_per_core for SpiNNaker,
            model_name for GeNN)
        """

        self.multiprocessor = multiprocessor


        label = self.check_multiproc(label)

        if label not in self.labels:
            self.labels.append(label)

        self.sim[label] = self.select_simulator(simulator_name)
        self.pynnx[label] = PyNNAL(self.sim[label])

        self.pynnx[label].setup(timestep, min_delay, per_sim_params)

        # self.description = description
        self.build_populations(description['populations'], label)

        self.build_projections(description['projections'], label)


    def select_simulator(self, sim_name):
        """Select proper PyNN backend --- util in __init__.py"""
        return backend_setup(sim_name)

    def build_populations(self, pop_desc, sim_label):
        """Generate all populations using the PyNN Abstraction Layer (pynnx) to
        avoid ugly code (at least here :P )"""

        pops = {}
        for label in pop_desc:
            _size = pop_desc[label]['size']
            _type = pop_desc[label]['type']
            _params = pop_desc[label]['params']

            pops[label] = self.pynnx[sim_label].Pop(_size, _type, _params, label)

            if 'record' in pop_desc[label]:
                self.record_ids[sim_label] = {}
                self.record_ids[sim_label][label] = pop_desc[label]['record']
                for rec in self.record_ids[sim_label][label]:
                    self.pynnx[sim_label].set_recording(pops[label], rec)

        self.populations[sim_label] = pops

    def build_projections(self, proj_desc, sim_label):
        """Generate all projections using the PyNN Abstraction Layer (pynnx) to
        avoid ugly code (at least here :P )"""
        projs = {}
        for src in proj_desc:
            projs[src] = {}
            for dst in proj_desc[src]:
                _source = self.populations[sim_label][src]
                _dest = self.populations[sim_label][dst]
                _conn = proj_desc[src][dst]['conn']
                _w = proj_desc[src][dst].get('weights', None)
                _d = proj_desc[src][dst].get('delays', None)
                _tgt = proj_desc[src][dst].get('target', 'excitatory')
                _stdp = proj_desc[src][dst].get('stdp', None)
                _lbl = proj_desc[src][dst].get('label', '{} to {}'.format(src, dst))
                _conn_p = proj_desc[src][dst].get('conn_params', {})

                projs[src][dst] = self.pynnx[sim_label].Proj(_source, _dest,
                                    _conn, _w, _d, _tgt, _stdp, _lbl, _conn_p)

        self.projections[sim_label] = projs

    def set_var(self, population, variable, value):
        self.pynnx.set_pop_attr(self.populations[population],
                                variable, value)

    def run(self, run_time, recordings=None, label=None):
        # sys.stderr.write("in run\n\n")
        # sys.stderr.flush()

        sim_label = self.check_multiproc(label)

        if recordings is not None and len(recordings):
            if sim_label not in self.record_ids:
                self.record_ids[sim_label] = {}

            for pop_label in recordings:
                if pop_label not in self.record_ids[sim_label]:
                    self.record_ids[sim_label][pop_label] = []
                pop = self.populations[sim_label][pop_label]

                for rec in recordings[pop_label]:
                    self.record_ids[sim_label][pop_label].append(rec)
                    self.pynnx[sim_label].set_recording(pop, rec)

        self.pynnx[sim_label].run(run_time)


    def get_records(self, label=None):
        # sys.stderr.write("in get_records\n\n")
        # sys.stderr.flush()

        sim_label = self.check_multiproc(label)

        if sim_label not in self.record_ids:
            raise Exception("No recordings were set before simulation")

        recs = {}
        for pop_label in self.record_ids[sim_label]:
            recs[pop_label] = {}
            for rec in self.record_ids[sim_label][pop_label]:
                # print("\n\n***********\n")
                # print("in PyNNServer.get_records ({}, {})".format(pop_label, rec))
                # print("\n***********\n")
                recs[pop_label][rec] = self.pynnx[sim_label].get_record(
                                            self.populations[sim_label][pop_label], rec)
        
        # print("\n\n+++++\n")
        # print("exiting PyNNServer.get_records")
        # for p in recs:
        #     for rt in recs[p]:
        #         for r in recs[p][rt]:
        #             print("{}, {}, {} =: {}, {}".\
        #                     format(p, rt, r, type(p), type(rt), type(r)))
                
        # print("\n+++++\n")
        return recs
    
    def get_weights(self, weights_to_get):
        _ws = {}
        return _ws


    def __del__(self):
        for label in self.labels:
            self.end(label)

    def end(self, label=None):
        sim_label = self.check_multiproc(label)
        try:
            self.pynnx[sim_label].end()
        except:
            pass

def main():
    Pyro4.Daemon.serveSimple(
            {
                NeuralNetworkServer: "spikevo.pynn_server"
            },
            ns = True, 
            verbose=True, 
            # host="mypynnserver"
        )

if __name__=="__main__":
    main()
