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

params = Parameters()
for i in range(20):
    params.add(f"a{i}", value=1, min=0)
params.add("steps", value=1, min=1)

prob = [0.29198528456392614, 0.1154059069301136, 0.024522486069846004, 0.22104755332176243, 0.34703876911435183]

def func(params, graph, data):
    i = 0
    for node in graph.values():
        for neighbor in range(len(node["neighbors"])):
            node["weights"][neighbor] = params[f"a{i}"].value
            i += 1

    walk = DTQW(graph)
    walk.simulate(int(params["steps"].value), "last")

    copy = walk._probabilities[0].copy()
    copy += [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    new_data = data.copy()
    new_data += [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    result = array(new_data) - array(copy)

    return result.flatten()

fit_result = minimize(func, params, args=(grapo, prob))

report_fit(fit_result)

i = 0
for node in grapo.values():
    for neighbor in range(len(node["neighbors"])):
        node["weights"][neighbor] = fit_result.params[f"a{i}"].value
        i += 1

a = DTQW(grapo)
a.simulate(int(fit_result.params["steps"].value), "last")
a.plotProbabilities()
err = [abs(a._probabilities[0][i] - prob[i]) for i in range(len(prob))]
print(err)


""" G = nx.Graph()

G.add_edge(0, 1, weight=0.6)
G.add_edge(0, 2, weight=0.2)
G.add_edge(2, 3, weight=0.1)
G.add_edge(2, 4, weight=0.7)
G.add_edge(2, 5, weight=0.9)
G.add_edge(0, 3, weight=0.3)
data = G.adjacency()
for node, neighbors in G.adjacency():
    print(f"Node {node} is connected to {list(neighbors.keys())}") """

""" dod = {

    0: {
        "neighbors" : [1, 2, 3, 4],
        "weights" : [1, 5, 3, 2]
    },

    1: {
        "neighbors" : [0, 2, 4],
        "weights" : [6, 3, 9]
    },

}

dod = {

    0: {
        1: {"weight": 1},
        2: {"weight": 5},
        3: {"weight": 3},
        4: {"weight": 2}
    },

    1: {
        0: {"weight": 6},
        2: {"weight": 3},
        4: {"weight": 9}
    }

}

a = {0: [1, 2], 1: [0, 3], 2: [0], 3: [1]}

G = nx.from_dict_of_dicts(dod)
for node, neighbors in G.adjacency():
    for value in neighbors.values():
        print(value["weight"]) """