from __future__ import (print_function,
                        # unicode_literals,
                        division)
from builtins import open, range, dict
import sys
import numpy as np

import copy

class Node(object):
    def __init__(self, pid, is_source=False, depth=0):
        self.id = pid
        self.is_source = is_source
        self.place = np.zeros(2, dtype='uint8')
        self.place_id = -1
        self.outputs = {}
        self.sub_link = {}
        self.parent = None
        self.depth = depth



    def dist2(self, node):
        return np.dot(self.place, node.place)

    def connect_to(self, node):
        self.outputs[node.id] = node
        node.parent = self
        if node.depth < (self.depth + 1):
            node.depth = self.depth + 1

class Graph(object):
    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.sources = {}
        self.inverse_sources = {}
        self.pops = {}
        self.width = 0
        self.height = 0
        self.link_count = 0

    def add(self, pop, is_source):
        # print(dir(pop))
        pid = pop.label

        if pid in self.nodes:
            sys.stderr.write("Population {} has a duplicate label, randomizing!\n".format(pid))
            sys.stderr.flush()
            pid = "{}_{}".format(pid, np.random.randint(100000, dtype='uint32'))
            sys.stderr.write("\tnew name {}\n\n".format(pid))
            sys.stderr.flush()
            # pop.label = pid

        self.pops[pid] = pop

        if is_source:
            self.sources[pid] = Node(pid, is_source=is_source, depth=0)
        else:
            self.nodes[pid] = Node(pid, is_source=is_source)

        return pid

    def plug(self, source_pop, sink_pop, w=1):
        if sink_pop.label not in self.nodes:
            raise Exception("Sink population {} has not been registered".format(sink_pop.label))

        elif source_pop.label in self.nodes:
            self.nodes[source_pop.label].connect_to(self.nodes[sink_pop.label])
            edge_dict = self.edges.get(source_pop.label, dict())
            edge_dict[sink_pop.label] = w
            self.edges[source_pop.label] = edge_dict

        elif source_pop.label in self.sources:
            self.inverse_sources[sink_pop.label] = source_pop.label
            self.sources[source_pop.label].connect_to(self.nodes[sink_pop.label])

        else:
            raise Exception("Source population {} has not been registered\n\nNodes: {}\n\nSources: {}".\
                        format(source_pop.label, self.nodes.keys(), self.sources.keys()))

        if self.height < self.nodes[sink_pop.label].depth:
            self.height = self.nodes[sink_pop.label].depth

        self.link_count += 1

    def clone(self):
        new_graph = Graph()
        new_graph.nodes = copy.deepcopy(self.nodes)
        new_graph.sources = copy.deepcopy(self.sources)
        new_graph.pops = self.pops
        new_graph.width = self.width
        new_graph.height = self.height

        return new_graph

    def evaluate(self, distances, mapping, subpop_weight=1.0):
        dist2 = 0.0
        subdist = 0.0
        for src in self.nodes:
            targets = self.nodes[src].outputs
            for tgt in targets:
                idx_src = self.nodes[src].place_id
                idx_tgt = self.nodes[tgt].place_id
                dist2 += distances[idx_src][idx_tgt]

            targets = self.nodes[src].sub_link
            for tgt in targets:
                idx_src = self.nodes[src].place_id
                idx_tgt = self.nodes[tgt].place_id
                subdist += distances[idx_src][idx_tgt]


        return dist2, (subdist * subpop_weight)

    def update_places(self, places):
        for _id in sorted(places):
            self.nodes[_id].place[:] = places[_id]


    def link_subpop(self, label0, label1):
        self.nodes[label0].sub_link[label1] = self.nodes[label1]
