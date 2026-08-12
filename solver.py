from ket import *
from ket.qulib.prepare import state as ket_state_prep
from numpy import array, ceil, floor, log2, sqrt, float64
import networkx as nx

from matplotlib import use as mplUse
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from lmfit import Parameters, minimize, report_fit

from discrete_walk import DTQW

import copy

grapo = {
    0: {
        "neighbors": [1, 2, 3, 4],
        "weights": [1, 1, 1, 1]
    },

    1: {
        "neighbors": [0, 2, 3, 4],
        "weights": [1, 1, 1, 1]
    },

    2: {
        "neighbors": [0, 1, 3, 4],
        "weights": [1, 1, 1, 1]
    },

    3: {
        "neighbors": [0, 1, 2, 4],
        "weights": [1, 1, 1, 1]
    },

    4: {
        "neighbors": [0, 1, 2, 3],
        "weights": [1, 1, 1, 1]
    }
}

class Solver:

    def __init__(self, graph : dict):

        self.original_graph = graph
        self.original_params = Parameters()

        for i in range(20):
            self.original_params.add(f"a{i}", value=1, min=0)
        self.original_params.add("steps", value=0, min=0)

    def func(self, params, graph, data):

        i = 0
        for node in graph.values():
            for neighbor in range(len(node["neighbors"])):
                node["weights"][neighbor] = params[f"a{i}"].value
                i += 1

        walk = DTQW(graph)
        walk.simulate(int(params["steps"].value), "last")

        appendix = [0 for _ in range( len(params) - len(graph) + 1)]

        copy = walk._probabilities[0].copy()
        copy += appendix

        new_data = data.copy()
        new_data += appendix

        result = array(new_data) - array(copy)

        return result.flatten()

    def run(self, steps : int, prob_list : list):

        fit_params = copy.deepcopy(self.original_params)
        fit_params["steps"].value = steps
        fit_params["steps"].min = steps

        fit_graph = copy.deepcopy(self.original_graph)

        fit_result = minimize(self.func, fit_params, args=(fit_graph, prob_list))
        report_fit(fit_result)

        walk = DTQW(fit_graph)
        walk.simulate(steps, "last")
        walk.plotProbabilities()

        err = [abs(walk._probabilities[0][i] - prob_list[i]) for i in range(len(prob_list))]
        print(err)

prob = [0.2793960352570557, 0.10794465769579394, 0.05543637917943149, 0.25872634630269153, 0.29849658156502745]

s = Solver(grapo)
s.run(10, prob)
#print(s.original_graph)
#print(s.fit_graph)