from ket import *
from ket.qulib.prepare import state as ket_state_prep
from numpy import array, ceil, floor, log2, sqrt, float64
from networkx import Graph
from matplotlib import use as mplUse
from matplotlib.colors import Normalize
from lmfit import Parameters, minimize, report_fit
from statistics import mean

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.linalg
import copy

from discrete_walk import DTQW

class Solver:

    def __createGraph(self, prob_list : list) -> list[list[float]]:
        G = nx.complete_graph(len(prob_list))
        graph = nx.adjacency_matrix(G)
        return graph.toarray().tolist()

    def __init__(self):
        self.result_graph = None
        self.ctqw_prob_list = None
        self.is_target_set = False

    def __fitFunc(self, params, graph, data):

        for i in range(len(graph)):
            for j in range(len(graph)):
                if (i != j):
                    graph[i][j] = params[f"w{i}{j}"].value
                else:
                    graph[i][j] = 0

        walk = DTQW(graph)
        walk.simulate(int(params["steps"].value), "last")

        appendix = [0 for _ in range( len(params) - len(graph) + 1)]

        copy = walk._probabilities[0].copy()
        copy += appendix

        new_data = data.copy()
        new_data += appendix

        result = array(new_data) - array(copy)

        return result.flatten()

    def setTarget(self, CTQW_graph : Graph, steps : float, amplitude: list[complex] = None):

        A = nx.adjacency_matrix(CTQW_graph).toarray()

        if amplitude == None:
            amplitude = [1/sqrt(len(A)) for _ in range(len(A))]

        H = -A
        t = steps
        U = scipy.linalg.expm(-1j * H * t)
        psi_0 = array(amplitude)
        psi_t = U @ psi_0
        self.ctqw_prob_list = np.abs(psi_t) ** 2
        self.is_target_set = True

    def solve(self, steps : int, prob_list : list = None, print_err : bool = False, print_fit_info : bool = False):

        if prob_list != None and self.is_target_set:
            raise ValueError("Target is already defined, don't pass a prob_list")

        if prob_list == None:
            if not self.is_target_set:
                raise ValueError("Target is not set, please pass a prob_list or define a target")
            prob_list = self.ctqw_prob_list.tolist()

        fit_graph = self.__createGraph(prob_list)

        fit_params = Parameters()
        for i in range(len(fit_graph)):
            for j in range(len(fit_graph)):
                if (i != j):
                    fit_params.add(f"w{i}{j}", value=fit_graph[i][j]*0.5, min=0, max=1)

        fit_params.add(f"steps", value=steps, min=steps)

        fit_result = minimize(self.__fitFunc, fit_params, args=(fit_graph, prob_list))
        if (print_fit_info):
            report_fit(fit_result)

        self.walk = DTQW(fit_graph)
        self.walk.simulate(steps, "last")
        self.walk.plotProbabilities()

        self.result_graph = fit_graph

        if (print_err):
            err = [abs(self.walk._probabilities[0][i] - prob_list[i]) for i in range(len(prob_list))]
            print(f"Maximal Error: {max(err)}")
            print(f"Average Error: {mean(err)}")
            print(f"Sum of Errors: {sum(err)}")

    def reset(self):
        self.result_graph = None
        self.ctqw_prob_list = None
        self.is_target_set = False

prob = [0.10256259, 0.01338814, 0.12367855, 0.01184509, 0.06701715, 0.00100341, 0.4620547, 0.21845037]

study_matrix = {
    0: [2, 7],
    1: [4],
    2: [0, 4, 6],
    3: [4, 5],
    4: [1, 2, 3],
    5: [3],
    6: [2, 7],
    7: [0, 6]
}

G = nx.from_dict_of_lists(study_matrix)

s = Solver()
#s.setTarget(G, 2, [1, 0, 0, 0, 0, 0, 0, 0])
s.solve(steps=1, prob_list=prob, print_err=True)