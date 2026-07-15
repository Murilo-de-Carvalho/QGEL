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

def func(params, graph, prob_list):
        i = 0
        for node in graph.values():
            for neighbor in range(len(node["neighbors"])):
                node["weights"][neighbor] = params[f"w{i}"].value
                i += 1

        walk = DTQW(graph)
        walk.simulate(int(params["steps"].value * 10), "last", starting_node=0)

        appendix = [0 for _ in params]

        copy = walk._probabilities[0].copy()
        copy += appendix.copy()
        new_data = prob_list.copy()
        new_data += appendix.copy()

        result = array(new_data) - array(copy)

        return result.flatten()

class Solver:

    def __getParams(self) -> None:
        self.params.add("steps", value=0.1, min=0.1)
        index = 0
        for key in self.graph.keys():
            for i in self.graph[key]["weights"]:
                self.params.add(f"w{index}", value=i, min=0)
                index += 1

    def __init__(self, graph : dict[int, dict[str, float]], prob_list : list[float]):
        self.graph = graph
        self.dw = DTQW(graph)
        self.params = Parameters()
        self.__getParams()
        self.prob_list = prob_list

    def minimize(self):
        fit_result = minimize(func, self.params, args=(self.graph, self.prob_list))
        report_fit(fit_result)
        i = 0
        for node in grapo.values():
            for neighbor in range(len(node["neighbors"])):
                node["weights"][neighbor] = fit_result.params[f"w{i}"].value
                i += 1

        b = DTQW(grapo)
        b.simulate(1, "all", starting_node=0)
        b.plotProbabilities()

prob = [0 for _ in range(5)]
prob[1] = 1/6
prob[2] = 2/6
prob[3] = 3/6

a = Solver(grapo, prob)
a.minimize()
print(a.params)