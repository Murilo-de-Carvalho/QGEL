from statistics import mean

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.linalg
from discrete_walk import DTQW
from ket import *
from lmfit import Parameters, minimize, report_fit
from networkx import Graph
from numpy import array, sqrt
from scipy import sparse


def my_callback(params, iteration, resid, *args, **kws):
    pass
    #print(f'Iter {iteration}, chi2 = {np.linalg.norm(resid)}')

class Solver:

    def __createGraph(self, prob_list : list) -> list[list[float]]:
        G: Graph = nx.complete_graph(len(prob_list))
        graph: sparse.sparse = nx.adjacency_matrix(G)
        return graph.toarray().tolist()

    def __init__(self):
        self.result_graph = None
        self.ctqw_prob_list = []  # ty: ignore[invalid-assignment]
        self.is_target_set = False

    def __fitFunc(self,params : dict, graph, data):

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
        return result

        lmbda = 0.001
        l1_penalties = sqrt(lmbda) * array([abs(value) for key, value in params.items() if key != "steps"])

        return np.concatenate([result, l1_penalties])

    def setTarget(self, CTQW_graph : Graph, steps : float, amplitude: list[complex] = []):  # noqa: B006

        A = nx.adjacency_matrix(CTQW_graph).toarray()

        if amplitude == []:
            amplitude = [1/sqrt(len(A)) for _ in range(len(A))]

        H = -A
        t = steps
        U = scipy.linalg.expm(-1j * H * t)
        psi_0 = array(amplitude, dtype=complex)
        psi_t = U @ psi_0
        self.ctqw_prob_list: np.ndarray = np.abs(psi_t) ** 2
        self.is_target_set = True

    def solve(self, steps : int, prob_list : list = [], print_err : bool = False, print_fit_info : bool = False):  # noqa: B006

        if prob_list != [] and self.is_target_set:
            raise ValueError("Target is already defined, don't pass a prob_list")

        if prob_list == []:
            if not self.is_target_set:
                raise ValueError("Target is not set, please pass a prob_list or define a target")
            prob_list = self.ctqw_prob_list.tolist()

        fit_graph = self.__createGraph(prob_list)

        fit_params = Parameters()
        for i in range(len(fit_graph)):
            for j in range(len(fit_graph)):
                if (i != j):
                    fit_params.add(f"w{i}{j}", value=fit_graph[i][j]*0.5, min=0, max=1)

        fit_params.add("steps", value=steps, vary=True)

        fit_result = minimize(self.__fitFunc, fit_params, args=(fit_graph, prob_list), iter_cb=my_callback, max_nfev=8000)
        if (print_fit_info):
            report_fit(fit_result)

        self.walk = DTQW(fit_graph)  # ty: ignore[invalid-argument-type]
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
        self.ctqw_prob_list = []  # ty: ignore[invalid-assignment]
        self.is_target_set = False

    def drawResult(self):

        G = nx.from_numpy_array(array(self.result_graph), create_using=nx.DiGraph)

        edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())
        pos = nx.spring_layout(G)

        nx.draw(
            G,
            pos,
            with_labels=True,
            node_size=400,
            node_color='black',
            font_size=13,
            font_color="white",
            edgelist=edges,
            edge_color=weights,
            width=3.0,
            edge_cmap=plt.cm.Purples,  # ty: ignore[unresolved-attribute]
            connectionstyle=f'arc3,rad={0.15}'
        )

        plt.show()

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
s.setTarget(G, 1.7, [1, 0, 0, 0, 0, 0, 0, 0])
s.solve(1, print_err=True, print_fit_info=True)
# TODO: dobrar params para adicionar parte imaginária e real, pegar módulo das duas e constuir o compelxo na fit func
"""
Err[1] = 1.0432293939572224e-09
Err[2] = 0.01311966645704642
Err[3] = 1.3947243855444349e-08
Err[4] = 0.004338559410773129
Err[5] = 0.0023804685375188148
Err[6] = 0.0002849151365401625
Err[7] = 0.0196086739586509
Err[8] = 0.009082453965216123
Err[9] = 0.03491729838626366
Err[10] = 0.0020074772440541212
"""

""" s = Solver()
s.setTarget(G, 1.7, [1, 0, 0, 0, 0, 0, 0, 0])
s.solve(1, print_err=True, print_fit_info=True)

matrix = s.result_graph
for i in range(8):
    for j in range(8):
        if matrix[i][j] < 1e-9:
            matrix[i][j] = 0

print(matrix)

d = DTQW(matrix)
d.simulate(1)
d.plotProbabilities() """
#s.drawResult()