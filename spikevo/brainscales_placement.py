from __future__ import (print_function,
                        # unicode_literals,
                        division)
from builtins import open, range, dict

import numpy as np
from deap import base
from deap import creator
from deap import tools
from deap import algorithms
import os
import matplotlib.pyplot as plt
from pprint import pprint


class WaferPlacer(object):
    def __init__(self, graph, wafer, constraints={}):
        self._graph = graph.clone()
        self._wafer = wafer.clone()
        self._constraints = constraints
        self._set_constraints()
        # self._place()



    def _set_constraints(self):
        pass



    # def _place(self, algorithm='simulated_annealing'):
    def _place(self, algorithm='genetic_algorithm'):
        if algorithm == 'simulated_annealing':
            return  self.simulated_annealing()
        elif algorithm == 'genetic_algorithm':
            return self.genetic_algorithm()
        else:
            return self.random_placement()

    def random_placement(self):
        width = self._graph.width + 2
        height = self._graph.height + 2

        clean_ids, clean_coords = self._wafer.available(width, height)
        IND_SIZE = len(self._graph.nodes)

        return np.random.choice(clean_ids, IND_SIZE, False)

    def genetic_algorithm(self):
        def parse_individual_to_graph(individual):
            keys = self._graph.nodes.keys()
            for i, hicann in enumerate(individual):
                # print(i, hicann)
                if hicann >= 384:
                    hicann = 383

                self._graph.nodes[keys[i]].place[:] = self._wafer.i2c(hicann)
                self._graph.nodes[keys[i]].place_id = hicann

        def plot_individual(individual):
            parse_individual_to_graph(individual)

            plc = self.places
            # pprint(plc)

            fig = plt.figure()
            ax = plt.subplot(1, 1, 1)

            for k in plc:
                color = 'r' if k.startswith('neurons') else 'b'
                plt.plot(plc[k][1], plc[k][0], 's', color=color)

            ax.set_xlim(-1, self._wafer._width + 1)
            ax.set_ylim(self._wafer._height + 1, -1)

            plt.show()

        def new_individual(indices, size):
            return np.random.choice(indices, size, False)

        def evaluate(individual):
            try:
                parse_individual_to_graph(individual)
            except:
                return 10.0**3, 10.0**3

            eval, sub_eval = self._graph.evaluate(self._wafer.distances, self._wafer.id2idx, subpop_weight=1.0)
            unique_error = len(individual) - len(np.unique(individual))
            # print(len(individual), len(np.unique(individual)), len(individual) - len(np.unique(individual)))
            # print(evaluation, unique_error)
            coords = np.array([list(self._wafer.i2c(i)) for i in individual])
            min_x, min_y = np.min(coords[:, 1]), np.min(coords[:, 0])
            max_x, max_y = np.max(coords[:, 1]), np.max(coords[:, 0])
            # area = (max_x - min_x) + (max_y - min_y)
            area = (max_x - min_x) * (max_y - min_y)

            # return eval, sub_eval
            return area, eval


        def make_unique(individual, indices):
            tmp = individual.copy()
            unique, counts = np.unique(tmp, return_counts=True)
            while len(unique) != len(tmp):
                for i, c in enumerate(counts):
                    if c > 1:
                        hicann = unique[i]
                        whr = np.where(tmp == hicann)[0]
                        for j, w in enumerate(whr):
                            # valid_indices = np.setdiff1d(indices, unique)
                            # tmp[w] = np.random.choice(valid_indices)
                            tmp[w] += j
                unique, counts = np.unique(tmp, return_counts=True)

            individual[:] = tmp

            return individual

        def explore_neighbourhood(individual, max_dr=1, max_dc=1):
            best_eval = np.sum(evaluate(individual))
            best_ind = individual.copy()
            tmp = individual.copy()
            for i, hicann in enumerate(individual):
                #somehow I manage to get hicann == 384!
                if hicann > 383:
                    hicann = 383

                row0, col0 = self._wafer.i2c(hicann)

                for dr in range(-max_dr, max_dr+1):
                    for dc in range(-max_dc, max_dc+1):
                        if dr == 0 and dc == 0:
                            continue

                        tmp[:] = individual
                        row, col = row0 + dr, col0 + dc
                        try:
                            new_hicann = self._wafer.c2i(row, col)
                        except:
                            continue

                        if new_hicann in individual:
                            continue

                        tmp[i] = new_hicann
                        eval = np.sum(evaluate(tmp))
                        if eval < best_eval:
                            best_eval = eval
                            best_ind[:] = tmp

            return best_ind


        def cxTwoPointCopy(ind1, ind2, indices):
            """Execute a two points crossover with copy on the input individuals. The
            copy is required because the slicing in numpy returns a view of the data,
            which leads to a self overwritting in the swap operation.
            """
            # np.random.seed()
            size = len(ind1)
            cxpoint1 = np.random.randint(1, size)
            cxpoint2 = np.random.randint(1, size - 1)
            if cxpoint2 >= cxpoint1:
                cxpoint2 += 1
            else:  # Swap the two cx points
                cxpoint1, cxpoint2 = cxpoint2, cxpoint1

            nind1 = ind1.copy()
            nind2 = ind2.copy()
            nind1[cxpoint1:cxpoint2] = ind2[cxpoint1:cxpoint2].copy()
            nind2[cxpoint1:cxpoint2] = ind1[cxpoint1:cxpoint2].copy()


            nind1[:] = make_unique(nind1, indices)
            nind2[:] = make_unique(nind2, indices)

            # print(np.sort(ind1), np.sort(ind2))
            nind1[:] = explore_neighbourhood(nind1)
            nind2[:] = explore_neighbourhood(nind2)
            # print(np.sort(ind1), np.sort(ind2))

            ind1[:] = nind1
            ind2[:] = nind2
            return ind1, ind2

        def mutIndividual(individual, indices, indpb):
            # np.random.seed()
            for i in range(len(individual)):
                if np.random.uniform(0, 1) < indpb:
                    valid_indices = np.setdiff1d(indices, individual)
                    individual[i] = np.random.choice(valid_indices)

                    # best_eval = np.sum(evaluate(individual))
                    # tmp = individual.copy()
                    # best_ind = individual.copy()
                    # for j in range(len(individual)):
                    #     if i == j:
                    #         continue
                    #     tmp[i], tmp[j] = tmp[j], tmp[i]
                    #     tmp_eval = np.sum(evaluate(tmp))
                    #     if tmp_eval < best_eval:
                    #         best_eval = tmp_eval
                    #         best_ind[:] = tmp
                    # individual[:] = tmp


            return individual,


        np.random.seed()

        width = self._graph.width + 2
        height = self._graph.height + 2

        clean_ids, clean_coords = self._wafer.available(width, height)
        print("clean_ids", clean_ids)
        print("clean_coords", clean_coords)
        creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
        creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)
        IND_SIZE = len(self._graph.nodes)

        toolbox = base.Toolbox()
        # np.random.choice args: (set from which to choose, sample size, with replacement? )
        toolbox.register("attr_indices", np.random.choice, clean_ids, IND_SIZE, False)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_indices)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("evaluate", evaluate)
        toolbox.register("mate", cxTwoPointCopy, indices=clean_ids)
        toolbox.register("mutate", mutIndividual, indices=clean_ids, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=10)
        # toolbox.register("select", tools.selBest)
        # toolbox.register("select", tools.selWorst)

        n_ind = 5 * len(self._graph.nodes)
        n_gen = 500
        pop = toolbox.population(n=n_ind)

        #find the best in population just to see differences with the final solution
        best_ind = pop[0].copy()
        best_eval = np.sum(evaluate(best_ind))
        for ind in pop:
            eval = np.sum(evaluate(ind))
            if eval < best_eval:
                best_eval = eval
                best_ind[:] = ind

        plot_individual(best_ind)
        #end find best

        hof = tools.HallOfFame(1, similar=np.array_equal)

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        pop1, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.5, ngen=n_gen, stats=stats,
                                        halloffame=hof)
        # pop1, log = algorithms.eaMuPlusLambda(pop, toolbox, mu=10, lambda_=100, cxpb=0.5, mutpb=0.5,
        #                  ngen=n_gen, stats=stats, halloffame=hof)
        # pop1, log = algorithms.eaMuCommaLambda(pop, toolbox, mu=50, lambda_=100, cxpb=0.2, mutpb=0.2,
        #                  ngen=n_gen, stats=stats, halloffame=hof)
        # print(pop1)
        # print(log)


        plot_individual(hof[0])
        pprint(np.unique(hof[0], return_counts=True))
        # pprint(self.places)
        os.sys.exit()



    @property
    def places(self):
        return {i: self._graph.nodes[i].place for i in sorted(self._graph.nodes)}


    def simulated_annealing(self):
        np.random.seed()
        def parse_individual_to_graph(placements):
            keys = self._graph.nodes.keys()
            for i, hicann in enumerate(placements):
                # print(i, hicann)
                if hicann >= 384:
                    hicann = 383

                self._graph.nodes[keys[i]].place[:] = self._wafer.i2c(hicann)
                self._graph.nodes[keys[i]].place_id = hicann

        def evaluate(placements):
            try:
                parse_individual_to_graph(placements)
            except:
                return 10.0**3, 10.0**3

            coords = np.array([list(self._wafer.i2c(i)) for i in placements])
            min_x, min_y = np.min(coords[:, 1]), np.min(coords[:, 0])
            max_x, max_y = np.max(coords[:, 1]), np.max(coords[:, 0])
            # area = (max_x - min_x) + (max_y - min_y)
            area = (max_x - min_x) * (max_y - min_y)

            eval, sub_eval = self._graph.evaluate(self._wafer.distances, self._wafer.id2idx, subpop_weight=1.0)
            unique_error = len(individual) - len(np.unique(individual))
            # print(len(individual), len(np.unique(individual)), len(individual) - len(np.unique(individual)))
            # print(evaluation, unique_error)
            return (area)# + eval + sub_eval)

        def run_steps(n_steps, max_distance, temperature, vertices, placements):
            n_accept = 0
            deltas = []
            for _ in range(n_steps):
                swapped, delta = _step(vertices, max_distance, temperature, placements)
                n_accept += (1 if swapped else 0)
                deltas.append(delta)
            # mean = np.mean(deltas)
            std  = np.std(deltas)
            cost = evaluate(placements)

            return n_accept, cost, std


        effort = 1.0
        max_r = max(self._wafer._height, self._wafer._width)
        width = self._graph.width + 2
        height = self._graph.height + 2

        nodes = [v for v in self._graph.nodes]
        num_nodes = len(nodes)
        num_edges = len(self._graph.edges)
        clean_ids, clean_coords = self._wafer.available(width, height)
        initial_placements = np.random.choice(clean_ids, num_nodes, False)
        _, _, cost_std = run_steps(num_nodes, max_r, 1e100, nodes, initial_placements)

        temperature = 20.0 * cost_std
        n_steps = max(1, int(effort * num_nodes**1.33))
        iter_count = 0
        current_cost = 0

        while temperature > ((0.005 * current_cost) / num_edges):
            n_accepted, current_cost, _ = run_steps(
                num_nodes, int(np.ceil(max_r)), temperature, nodes, initial_placements)

            r_accepted = n_accepted / float(n_steps)

            if current_cost == 0:
                break

            if r_accept > 0.96:
                alpha = 0.5
            elif r_accept > 0.8:
                alpha = 0.9
            elif r_accept > 0.15:
                alpha = 0.95
            else:
                alpha = 0.8

            temperature = alpha * temperature

            max_r *= 1.0 - 0.44 + r_accept
            max_r = min(max(max_r, 1.0),
                        max(self._wafer._width, self._wafer._height))

            iter_count += n_steps

