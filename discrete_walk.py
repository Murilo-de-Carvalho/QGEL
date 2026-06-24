from ket import *
from ket.qulib.prepare import state as ket_state_prep
from numpy import array, ceil, floor, log2, sqrt, float64
from numpy.linalg import norm
import networkx as nx

from matplotlib import use as mplUse
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

class DTQW:

    def __countNumOfNodes(self) -> None:
        num = max(self._adjacency_list)
        for value in self._adjacency_list.values():
            temp = max(value["neighbors"])
            if temp > num:
                num = temp
        self._num_nodes = num + 1 #starts at zero

    def __isListOfLists(self, var) -> bool:
        return ( isinstance(var, list) and isinstance(var[0], list) )

    def __isDictOfLists(self, var) -> bool:
        return ( isinstance(var, dict) and isinstance(next(iter(var.values()), None), list) )

    def __isDictOfDicts(self, var) -> bool:
        return ( isinstance(var, dict) and isinstance(next(iter(var.values()), None), dict) )

    def __handleListOfLists(self, graph) -> None:
        self._networkx_graph = nx.from_numpy_array(array(graph))
        self._num_nodes = len(graph)
        for i in range(self._num_nodes):
            self._adjacency_list[i] = {}
            self._adjacency_list[i]["neighbors"] = []
            self._adjacency_list[i]["weights"] = []
            for j in range(self._num_nodes):
                if graph[i][j]:
                    self._adjacency_list[i]["neighbors"].append(j)
                    self._adjacency_list[i]["weights"].append(graph[i][j])

    def __handleDictOfLists(self, graph) -> None:
        self._networkx_graph = nx.from_dict_of_lists(graph)
        for key in graph.keys():
            self._adjacency_list[key] = {}
            self._adjacency_list[key]["neighbors"] = graph[key]
            self._adjacency_list[key]["weights"] = [1 for _ in range(len(graph[key]))]
        self.__countNumOfNodes()

    def __handleDictOfDicts(self, graph) -> None:
        self._networkx_graph = nx.Graph()
        self._adjacency_list = graph
        self.__countNumOfNodes()
        for key in graph.keys():
            self._networkx_graph.add_edges_from(list((key, graph[key]["neighbors"][j], {"weight": graph[key]["weights"][j]}) for j in range(len(graph[key]["weights"]))))  

    def __handleNxGraph(self, graph) -> None:
        self._networkx_graph = graph
        for node, neighbors in graph.adjacency():
            self._adjacency_list[node] = {}
            self._adjacency_list[node]["neighbors"] = list(neighbors.keys())

            self._adjacency_list[node]["weights"] = []
            for value in neighbors.values():
                if value == {}:
                    self._adjacency_list[node]["weights"].append(1)
                else:
                    self._adjacency_list[node]["weights"].append(value["weight"])

        self.__countNumOfNodes()

    def __init__(self, graph : list[list[int]] | dict[int, list[int]] | dict[int, dict[str, float]] | nx.Graph):

        self._adjacency_list = {}
        self._num_nodes : int = 0
        self._networkx_graph : nx.Graph = None

        if self.__isListOfLists(graph):
            self.__handleListOfLists(graph)

        elif self.__isDictOfLists(graph):
            self.__handleDictOfLists(graph)

        elif self.__isDictOfDicts(graph):
            self.__handleDictOfDicts(graph)

        elif type(graph) == nx.Graph:
            self.__handleNxGraph(graph)

        else:
            raise TypeError("Type of graph not supported")

        self._num_qubits_nodes = int(ceil(log2(self._num_nodes)))
        self._num_qubits_total = 2 * self._num_qubits_nodes

        self._proc = Process()
        self._all_qubits = self._proc.alloc(self._num_qubits_total)
        self._first_node_qubits = self._all_qubits[:self._num_qubits_nodes]
        self._second_node_qubits = self._all_qubits[self._num_qubits_nodes:]

        self._probabilities : list[list[float]] = []
        self._steps : int = 0

    def __getAmplitudeOfNode(self, node : int) -> list[complex]:
        magnitude = norm(self._adjacency_list[node]["weights"])
        normalized_weights = self._adjacency_list[node]["weights"].copy()
        normalized_weights /= magnitude
        amplitude = [0 for _ in range(2 ** self._num_qubits_nodes)]
        for i in range(len(normalized_weights)):
            amplitude[self._adjacency_list[node]["neighbors"][i]] = normalized_weights[i]
        return amplitude

    def __getState(self, node : int) -> list[int]:
        binary = bin(node)[2:].zfill(self._num_qubits_nodes)
        return [int(binary[i]) for i in range(len(binary))]

    def __prepareCircuit(self) -> None:

        # Selecting fist node
        amplitudes = [1/sqrt(self._num_nodes) if i < self._num_nodes else 0 for i in range(2 ** self._num_qubits_nodes)]
        ket_state_prep(amplitudes, self._first_node_qubits)

        # Selecting second node based on the first
        for node in self._adjacency_list.keys():

            amps = self.__getAmplitudeOfNode(node)
            state = self.__getState(node)

            with control(self._first_node_qubits, state):
                ket_state_prep(amps, self._second_node_qubits)

    def __prepareCircuitFromStartingNode(self, node : int) -> None:

        amplitudes = [1 if i == node else 0 for i in range(2 ** self._num_qubits_nodes)]
        ket_state_prep(amplitudes, self._first_node_qubits)

        amps = self.__getAmplitudeOfNode(node)
        state = self.__getState(node)

        with control(self._first_node_qubits, state):
            ket_state_prep(amps, self._second_node_qubits)

    def __coin(self, node : int) -> None:

        amplitude = self.__getAmplitudeOfNode(node)
        state = self.__getState(node)

        with control(self._first_node_qubits, state):

            adj(ket_state_prep)(amplitude, self._second_node_qubits)

            X(self._second_node_qubits)
            H(self._second_node_qubits[-1])
            ctrl(self._second_node_qubits[:-1], X)(self._second_node_qubits[-1])
            H(self._second_node_qubits[-1])
            X(self._second_node_qubits)

            ket_state_prep(amplitude, self._second_node_qubits)

    def __shift(self) -> None:
        for i in range(self._num_qubits_nodes):
            SWAP(self._first_node_qubits[i], self._second_node_qubits[i])

    def __getProbabilities(self) -> None:

        dump_dict = dump(self._all_qubits).get()

        for i in dump_dict:
            dump_dict[i] = dump_dict[i] ** 2

        data = [0 for _ in range(self._num_nodes)]

        for key in dump_dict.keys():
            binary_total = bin(key)[2:].zfill(self._num_qubits_total)
            binary_node = binary_total[:self._num_qubits_nodes]
            data[int(binary_node, 2)] += abs(dump_dict[key])

        self._probabilities.append(data)

    def simulate(
        self,
        steps : int,
        register_probabilities : str = "last", # "all", "last" or "none"
        starting_node : int = -1,
        state_prep_list : list[complex] = []
    ) -> None:

        if steps < 1:
            raise ValueError("steps must be a positive integer")

        if register_probabilities != "last" and register_probabilities != "all" and register_probabilities != "none":
            raise ValueError("register_probabilities must be: 'all', 'last' or 'none'")

        if state_prep_list != [] and starting_node != -1:
            raise ValueError("starting_node and state_prep_list are mutually exclusive, please choose one")

        # Decided to simply ignore negative numbers, might change it in the future
        elif starting_node >= 0:
            if starting_node > self._num_nodes-1:
                raise ValueError(f"starting_node must be between 0 and {self._num_nodes-1}")
            self.__prepareCircuitFromStartingNode(starting_node)

        elif state_prep_list != []:
            if len(state_prep_list) != 2 ** (self._num_qubits_total):
                raise ValueError(f"state_prep_list must have {2 ** (self._num_qubits_total)} elements, one for each possible binary state")
            ket_state_prep(state_prep_list, self._all_qubits)

        else:
            self.__prepareCircuit()

        self._steps = steps

        for _ in range(steps):

            for node in self._adjacency_list.keys():
                self.__coin(node)

            self.__shift()

            if register_probabilities == "all":
                self.__getProbabilities()

        if register_probabilities == "last":
            self.__getProbabilities()

    def plotProbabilities(
        self,
        projection : str = "2d",
        nodes : list[int] = []
    ) -> None:

        if self._probabilities == []:
            raise ValueError("No probabilities available")

        if len(self._probabilities) == 1: # In case of only registering the final probability
            fig, ax = plt.subplots()
            ax.bar(range(self._num_nodes), self._probabilities[0])

            ax.set_xlabel("nodes")
            ax.set_xticks(list(range(self._num_nodes)))

            ax.set_ylabel("probability")

        else:

            nodes.sort()
            nodes = list(set(nodes)) # removing duplicates

            if projection.lower() == "3d":

                x = []
                y = []
                
                if nodes == []:
                    z_size = self._num_nodes * self._steps
                else:
                    z_size = len(nodes) * self._steps
                
                z = [0 for i in range(z_size)]

                dx = [0.5 for i in range(z_size)]
                dy = [0.5 for i in range(z_size)]
                dz = []

                fig, ax = plt.subplots(subplot_kw={"projection" : "3d"})
                for i in range(self._steps):
                    if nodes == []:
                        x += [j - 0.25 for j in range(self._num_nodes)]
                        y += [i + 0.75 for k in range(self._num_nodes)]
                        dz += self._probabilities[i]
                    else:
                        x += [j - 0.25 for j in range(len(nodes))]
                        y += [i + 0.75 for k in nodes]
                        temp = [0 for _ in range(len(nodes))]
                        for j in range(len(nodes)):
                            temp[j] = self._probabilities[i][nodes[j]]
                        dz += temp.copy()

                print(x)

                ax.set_xlabel("nodes")
                if nodes == []:
                    ax.set_xticks(list(range(self._num_nodes)))
                else:
                    ax.set_xticks(list(range(len(nodes))), nodes)

                ax.set_ylabel("steps")
                ax.set_yticks(list(range(1, self._steps + 1)))

                ax.set_zlabel("probability")

                cmap = cm.inferno
                norm = Normalize(vmin=min(dz), vmax=max(dz))
                colors = cmap(norm(dz))

                ax.bar3d(x, y, z, dx, dy, dz, shade=True, color=colors, zsort='max')


            if projection.lower() == "2d":

                if len(nodes) > 10:
                    raise ValueError("This type of plotting only allows 10 nodes visible at once for clarity")

                if nodes == [] and self._num_nodes > 10:
                    raise ValueError("There are more than 10 nodes on this graph, please specify at maximum 10 nodes to view at a time")

                colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k', 'tab:purple', 'tab:orange', 'tab:pink']
                used_colors = 0

                fig, ax = plt.subplots()

                ax.set_xlabel("steps")
                ax.set_xticks(list(range(1, self._steps + 1)))

                ax.set_ylabel("probability")

                x = [i+1 for i in range(self._steps)]
                zeroes = [0 for i in range(self._steps)]

                plots = []

                for i in range(self._num_nodes):
                    y = []

                    for j in range(self._steps):

                        if (nodes != [] and i in nodes) or (nodes == []):
                            y.append(self._probabilities[j][i])

                    if (y != []) and (nodes != [] or y != zeroes):
                        temp, = ax.plot(x, y, c=colors[used_colors], marker='o', linestyle="-.", label="Node " + str(i))
                        used_colors += 1
                        plots.append(temp)

                ax.legend(handles=plots)

        """ if print_connection_mapping:
            print("\n\t|NODE_1> <---[CONNECTION_ID]---> |NODE_2>\n")
            for i in range(self.num_connections):
                print(f"|{self._connections[i].node_1}> <---[{i} / {bin(i)[2:].zfill(self.num_qubits_connections)}]---> |{self._connections[i].node_2}>") """

        plt.show()

    def draw(self, show_labels : bool = True) -> None:
        nx.draw(self._networkx_graph, with_labels=show_labels, node_size=400, font_size=13, node_color="black", font_color="white")
        plt.show()

    def reset(self) -> None:

        self._probabilities : list[list[float]] = []
        self._steps = 0

        self._proc = Process()
        self._all_qubits = self._proc.alloc(self._num_qubits_total)
        self._first_node_qubits = self._all_qubits[:self._num_qubits_nodes]
        self._second_node_qubits = self._all_qubits[self._num_qubits_nodes:]

if __name__ == "__main__":

    study_matrix = [
        [0, 0, 1, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 1, 1, 0, 0],
        [0, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 1, 0]
    ]

    simple_4x41 = [
        [0, 1, 1, 1],
        [1, 0, 1, 0],
        [1, 1, 0, 0],
        [1, 0, 0, 0]
    ]

    lattice_5x5 = [
        [0, 1, 0, 1, 0],
        [1, 0, 1, 0, 1],
        [0, 1, 0, 1, 0],
        [1, 0, 1, 0, 1],
        [0, 1, 0, 1, 0]
    ]

    connected = [
        [0, 1, 1, 1, 1],
        [1, 0, 1, 1, 1],
        [1, 1, 0, 1, 1],
        [1, 1, 1, 0, 1],
        [1, 1, 1, 1, 0]
    ]

    bipart = [
        [0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 1, 1],
        [0, 0, 0, 1, 0, 1],
        [1, 0, 1, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0]
    ]

    cicle5x5 = [
        [0, 1, 0, 0, 1],
        [1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1],
        [1, 0, 0, 1, 0]
    ]

    cicle4x4 = [
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0]
    ]

    simple_4x4 = [
        [0, 1, 0, 0],
        [1, 0, 1, 1],
        [0, 1, 0, 1],
        [0, 1, 1, 0]
    ]

    simple_dict = {
        0: [1, 2, 3],
        1 : [0, 2, 3],
        2 : [0, 1],
        3: [0, 1]
    }

    G = nx.gnp_random_graph(100, 0.3)
    print("Nodes:", G.number_of_nodes(), "Edges:", G.number_of_edges())

    example = DTQW(graph=G)
    example.simulate(steps=1, register_probabilities="none")
    #example.plotProbabilities()
    #example.draw()
    #print(example._adjacency_list)