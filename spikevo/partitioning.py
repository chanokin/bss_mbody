from __future__ import (print_function,
                        # unicode_literals,
                        division)
from future.builtins import open, range, dict

import copy
import numpy as np
import numbers
from . import *
from .image_input import NestImagePopulation
from .wafer import Wafer as WAL
from .graph import Graph, Node
# from .brainscales_placement import *

import os

class SplitPop(object):
    pass


class SplitArrayPopulation(SplitPop):
    """
        When using the BrainScaleS toolchain we faced some problems with the
        partition and place-and-route algorithms. The initial problem is
        that the tools break when using certain connectivity (e.g. populations
        greater than 200 with one-to-one connectors). This class attempts to
        avoid the problem by partitionining before the toolchain requires it.
    """

    ### This seems like reinventing the wheel and I shouldn't have to!
    def __init__(self, pynnal, size, cell_class, params, label=None, shape=None,
                 max_sub_size=1):
        """
            pynnal: PyNN Abstraction Layer instance, we use it to avoid re-coding
                different cases for different PyNN versions while creating
                the populations.
            size: original size of the population
            cell_class: original PyNN cell type of the population
            params: original PyNN parameters for the given cell type
            label: original population label/name
            shape: shape of the original population (currently only 1D supported)
            max_sub_size: size of the sub-populations to be created
        """
        self.pynnal = pynnal
        self.size = size
        self.cell_class = cell_class
        self.params = params
        if label is None:
            self.label = "SplitArrayPopulation ({:05d})".format(
                np.random.randint(0, 99999))
        else:
            self.label = label

        if shape is None:
            self.shape = (size,)  # tuple expressing grid size, 1D by default
        else:
            assert np.prod(shape) == size, \
                "Total number of neurons should equal grid dimensions"
            self.shape = shape

        ### TODO: this will likely change if shape is not 1D
        self.max_sub_size = max_sub_size
        self.n_sub_pops = calc_n_part(self.size, self.max_sub_size)

        ### actually do the partitioning
        self.partition()

    def partition(self):
        if len(self.shape) == 1:
            pops = []
            count = 0
            template = " - sub %%0%dd" % int(np.ceil(self.n_sub_pops/10.0)+1)
            for i in range(self.n_sub_pops):
                size = min(self.max_sub_size, self.size - count)
                ids = np.arange(count, count + size)
                count += self.max_sub_size
                label = self.label + (template % (i + 1))
                if self.cell_class.__name__.lower() == 'spikesourcearray':
                    if isinstance(self.params['spike_times'][0], list):
                        params = {'spike_times': self.params['spike_times'][i]}
                # print(ids, label)
                pops.append({
                    'ids': ids,
                    'pop': self.pynnal.Pop(size, self.cell_class, params, label=label),
                    'label': label,
                })

        ### TODO: deal with 2D, 3D!
        self._populations = pops

    def record(self, recordable):
        for pop in self._populations:
            self.pynnal.set_recording(pop['pop'], recordable)




class SplitPopulation(SplitPop):
    """
        When using the BrainScaleS toolchain we faced some problems with the
        partition and place-and-route algorithms. The initial problem is
        that the tools break when using certain connectivity (e.g. populations
        greater than 200 with one-to-one connectors). This class attempts to
        avoid the problem by partitionining before the toolchain requires it.
    """

    ### This seems like reinventing the wheel and I shouldn't have to!
    def __init__(self, pynnal, size, cell_class, params, label=None, shape=None,
                 max_sub_size=BSS_MAX_SUBPOP_SIZE):
        """
            pynnal: PyNN Abstraction Layer instance, we use it to avoid re-coding
                different cases for different PyNN versions while creating
                the populations.
            size: original size of the population
            cell_class: original PyNN cell type of the population
            params: original PyNN parameters for the given cell type
            label: original population label/name
            shape: shape of the original population (currently only 1D supported)
            max_sub_size: size of the sub-populations to be created
        """
        self.pynnal = pynnal
        self.size = size
        self.cell_class = cell_class
        self.params = params
        if label is None:
            self.label = "SplitPopulation ({:05d})".format(
                np.random.randint(0, 99999))
        else:
            self.label = label

        if shape is None:
            self.shape = (size,)  # tuple expressing grid size, 1D by default
        else:
            assert np.prod(shape) == size, \
                "Total number of neurons should equal grid dimensions"
            self.shape = shape

        ### TODO: this will likely change if shape is not 1D
        self.max_sub_size = max_sub_size
        self.n_sub_pops = calc_n_part(self.size, self.max_sub_size)

        ### actually do the partitioning
        self.partition()

    def partition(self):
        if len(self.shape) == 1:
            pops = []
            count = 0
            template = " - sub %%0%dd" % int(np.ceil(self.n_sub_pops/10.0)+1)
            for i in range(self.n_sub_pops):
                size = min(self.max_sub_size, self.size - count)
                ids = np.arange(count, count + size)
                count += self.max_sub_size
                label = self.label + (template % (i + 1))
                pops.append({
                    'ids': ids,
                    'pop': self.pynnal.Pop(size, self.cell_class, self.params, label),
                    'label': label,
                })
                if i > 0:
                    self.pynnal._graph.link_subpop(pops[i - 1]['label'], pops[i]['label'])
                    self.pynnal._graph.link_subpop(pops[i]['label'], pops[i - 1]['label'])
                if i > 1:
                    self.pynnal._graph.link_subpop(pops[i - 2]['label'], pops[i]['label'])
                    self.pynnal._graph.link_subpop(pops[i]['label'], pops[i - 2]['label'])

                # if i > 2:
                #     self.pynnal._graph.link_subpop(pops[i - 3]['label'], pops[i]['label'])

        ### TODO: deal with 2D, 3D!
        self._populations = pops

    def record(self, recordable, to_file=False):
        for pop in self._populations:
            self.pynnal.set_recording(pop['pop'], recordable, to_file=to_file)


class SplitProjection(object):
    """
    Since we had to pre-partition the populations, now we need to split-up
    the projections as well.
    """

    ### This seems like reinventing the wheel and I shouldn't have to!
    def __init__(self, pynnal, source_pop, dest_pop, conn_class, weights, delays,
                 target='excitatory', stdp=None, label=None, conn_params={},
                digital_weights=None):
        self.digital_weights = digital_weights
        self.pynnal = pynnal
        self.source = source_pop
        self.destination = dest_pop
        self.conn_class = conn_class
        self.weights = weights
        self.w_min = np.min(np.abs(weights))
        self.w_max = np.max(np.abs(weights))
        self.delays = delays
        self.target = target
        self.stdp = stdp
        self.conn_params = conn_params
        self._projections = None
        if label is None:
            self.label = 'SplitProjection from {} to {}'.format(
                self.source.label, self.destination.label)
        else:
            self.label = label

        self.partition()

    def partition(self):
        src = self.source
        dst = self.destination
        conn, params = self.conn_class, self.conn_params
        w, d, tgt = self.weights, self.delays, self.target
        stdp = self.stdp

        if isinstance(src, SplitPopulation) or isinstance(src, SplitArrayPopulation):
            pres = src._populations
        else:
            pres = [{'ids': np.arange(src.size), 'pop': src, 'label': src.label}]

        if isinstance(dst, SplitPopulation) or isinstance(dst, SplitArrayPopulation):
            posts = dst._populations
        else:
            posts = [{'ids': np.arange(dst.size), 'pop': dst, 'label': dst.label}]

        projs = {}
        for src_part in pres:
            pre_ids, pre, pre_label = src_part['ids'], src_part['pop'], src_part['label']
            src_prjs = projs.get(pre_label, {})

            for dst_part in posts:
                post_ids, post, post_label = dst_part['ids'], dst_part['pop'], dst_part['label']
                lbl = '{} sub {} - {}'.format(self.label, pre_label, post_label)

                proj = self._proj(src_part, dst_part, conn, w, d,
                                  tgt, params, lbl, stdp)

                if proj is None:
                    continue

                src_prjs[post_label] = {'ids': {'pre': pre_ids, 'post': post_ids},
                                        'proj': proj}

            projs[pre_label] = src_prjs

        self._projections = projs

    def _proj(self, pre, post, conn, w, d, tgt, params, lbl, stdp=None):
        cname = conn if is_string(conn) else conn.__name__
        if cname.startswith('FromList'):
            return self.from_list_connector(pre, post, conn, w, d, tgt, params, lbl, stdp=None)
        elif cname.startswith('OneToOne'):
            return self.one_to_one_connector(pre, post, conn, w, d, tgt, params, lbl, stdp=None)
        elif cname.startswith('AllToAll'):
            return self.all_to_all_connector(pre, post, conn, w, d, tgt, params, lbl, stdp=None)
        elif cname.startswith('FixedProbability'):
            return self.stats_connector(pre, post, conn, w, d, tgt, params, lbl, stdp=None)
        else:
            raise Exception("unsupported connection for splitting")
            # return None

    def stats_connector(self, pre, post, conn, w, d, tgt, params, lbl, stdp=None):
        pynnal = self.pynnal
        param_copy = copy.copy(params)
        # print(pre['ids'])
        # print(post['ids'])
        # print(conn)
        # print(param_copy)
        return pynnal.Proj(pre['pop'], post['pop'], conn, w, d,
                           target=tgt, stdp=stdp, label=lbl, conn_params=param_copy,
                           digital_weights=self.digital_weights)

    def _sample(self, v, num):
        if np.isscalar(v):
            return np.full(num, v)
        elif isinstance(v, self.pynnal.sim.RandomDistribution):
            return v.next(n=num)

    def one_to_one_connector(self, pre, post, conn, w, d, tgt, params, lbl, stdp=None):
        pynnal = self.pynnal

        if len(pre['ids']) == len(post['ids']):
            if pre['ids'][0] != post['ids'][0] or pre['ids'][-1] != post['ids'][-1]:
                return None
            else:
                return pynnal.Proj(pre['pop'], post['pop'], conn, w, d,
                        target=tgt, stdp=stdp, label=lbl, conn_params=params,
                        digital_weights=self.digital_weights)
        else:
            indices = np.intersect1d(pre['ids'], post['ids'])

            if len(indices) == 0:
                return None
            pre_map = {idx: np.where(pre['ids'] == idx)[0][0] for idx in indices}
            post_map = {idx: np.where(post['ids'] == idx)[0][0] for idx in indices}
            ws = self._sample(w, len(indices))
            ds = self._sample(d, len(indices))

            params['conn_list'] = [(pre_map[idx], post_map[idx], ws[i], ds[i]) \
                                                        for i, idx in enumerate(indices)]

            return pynnal.Proj(pre['pop'], post['pop'], 'FromListConnector', None, None,
                               target=tgt, stdp=stdp, label=lbl, conn_params=params,
                               digital_weights=self.digital_weights)

    def all_to_all_connector(self, pre, post, conn, w, d, tgt, params, lbl, stdp=None):
        return self.stats_connector(pre, post, conn, w, d, tgt, params, lbl, stdp,                  digital_weights=self.digital_weights)

    def from_list_connector(self, pre, post, conn, w, d, tgt, params, lbl, stdp=None):
        pynnal = self.pynnal
        clist = params['conn_list']
        if isinstance(clist, list):
            clist = np.array(clist)

        whr = np.where(np.intersect1d(
            np.intersect1d(clist[:, 0], pre['ids'])[0],
            np.intersect1d(clist[:, 1], post['ids'])[0]))[0]
        cp = {'conn_list': clist[whr, :]}

        return pynnal.Proj(pre['pop'], post['pop'], conn, w[whr], d[whr],
                           target=tgt, stdp=stdp, label=lbl, conn_params=cp,
                           digital_weights=self.digital_weights)

    def getWeights(self, format='array'):
        pynnal = self.pynnal
        mtx = np.ones((self.source.size, self.destination.size)) * np.inf
        for row in self._projections:
            for col in self._projections[row]:
                part = self._projections[row][col]
                pre_ids = part['ids']['pre']

                r0, rN = np.min(pre_ids), np.max(pre_ids) + 1
                
                post_ids = part['ids']['post']
                c0, cN = np.min(post_ids), np.max(post_ids) + 1

                weights = pynnal.get_weights(part['proj'], format=format)
                mtx[r0:rN, c0:cN] = weights

        return mtx
