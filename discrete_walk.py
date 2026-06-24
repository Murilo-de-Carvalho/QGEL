from ket import *
from ket.qulib.prepare import state as ket_state_prep
from numpy import array, ceil, floor, log2, sqrt, float64
import networkx as nx

from matplotlib import use as mplUse
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

class Node:
    def __init__(self):
        self.degree = 0
        self.connected_nodes = []
        self.connection_ids = []

class Connection:
    def __init__(self, node_1 : int, node_2 : int):
        self.node_1 = node_1
        self.node_2 = node_2

class DTQW:

    def __init__(self, graph : list[list[int]] | nx.Graph):

        if type(graph) == nx.Graph:
            self.networkx_graph = graph
            graph = nx.adjacency_matrix(graph).toarray()

        else:
            self.networkx_graph = nx.from_numpy_array(array(graph))

        self.adj_matrix = graph
        self.num_nodes = len(self.adj_matrix)
        self._nodes = [Node() for _ in range(self.num_nodes)] # Initializing the node list
        self._connections = []
        self.num_connections = 0

        for i in range(self.num_nodes):
            for j in range(i, self.num_nodes):
                if (self.adj_matrix[i][j]):

                    self._nodes[i].degree += 1
                    self._nodes[j].degree += 1

                    self._nodes[i].connected_nodes.append(j)
                    self._nodes[j].connected_nodes.append(i)

                    self._nodes[i].connection_ids.append(self.num_connections)
                    self._nodes[j].connection_ids.append(self.num_connections)

                    self._connections.append(Connection(i, j))

                    self.num_connections += 1
        
        self.num_qubits_nodes = int(ceil(log2(self.num_nodes)))

        self.num_qubits_connections = int(ceil(log2(self.num_connections)))

        self.num_qubits_total = self.num_qubits_nodes + self.num_qubits_connections

        self.proc = Process()
        self.all_qubits = self.proc.alloc(self.num_qubits_total)
        self.node_qubits = self.all_qubits[:self.num_qubits_nodes]
        self.connection_qubits = self.all_qubits[self.num_qubits_nodes:]

        self.probabilities : list[list[float]] = []
        self.steps = 0

    def __prepareCircuit(self) -> None:

        num_possibilities = 2 ** self.num_qubits_total
        amplitudes = [0 for _ in range(num_possibilities)]
        inverse_of_sqrt_num_nodes = 1/sqrt(self.num_nodes)

        for node in range(self.num_nodes):
            inverse_of_sqrt_degree = 1/sqrt(self._nodes[node].degree)

            binary_node = bin(node)[2:].zfill(self.num_qubits_nodes)

            for connection_id in self._nodes[node].connection_ids:

                binary_connection = bin(connection_id)[2:].zfill(self.num_qubits_connections)

                state = binary_node + binary_connection

                amplitudes[int(state, 2)] = inverse_of_sqrt_degree * inverse_of_sqrt_num_nodes

        ket_state_prep(amplitudes, self.all_qubits)

    def __prepareCircuitFromStartingNode(self, node : int) -> None:

        num_possibilities = 2 ** self.num_qubits_total
        amplitudes = [0 for _ in range(num_possibilities)]

        inverse_of_sqrt_degree = 1/sqrt(self._nodes[node].degree)

        for connection_id in self._nodes[node].connection_ids:

            binary_node = bin(node)[2:].zfill(self.num_qubits_nodes)
            binary_connection = bin(connection_id)[2:].zfill(self.num_qubits_connections)

            state = binary_node + binary_connection

            amplitudes[int(state, 2)] = inverse_of_sqrt_degree
        
        ket_state_prep(amplitudes, self.all_qubits)

    def __getAmplitudeOfNode(self, node : int) -> list[float]:

        amplitudes = [0 for _ in range(2 ** self.num_qubits_connections)]

        inverse_of_sqrt_degree = 1/sqrt(self._nodes[node].degree)

        for connection_id in self._nodes[node].connection_ids:
            amplitudes[connection_id] = inverse_of_sqrt_degree

        return amplitudes

    def __getProbabilities(self):

        dump_dict = dump(self.all_qubits).get()

        for i in dump_dict:
            dump_dict[i] = dump_dict[i] ** 2

        data = [0 for _ in range(self.num_nodes)]

        for key in dump_dict.keys():
            binary_total = bin(key)[2:].zfill(self.num_qubits_total)
            binary_node = binary_total[:self.num_qubits_nodes]
            data[int(binary_node, 2)] += abs(dump_dict[key])

        self.probabilities.append(data)

    def __coin(self, node : int) -> None:

        binary_node = bin(node)[2:].zfill(self.num_qubits_nodes)

        amplitude = self.__getAmplitudeOfNode(node)

        state = [int(binary_node[i]) for i in range(len(binary_node))]
        with control(self.node_qubits, state):

            adj(ket_state_prep)(amplitude, self.connection_qubits)

            X(self.connection_qubits)
            H(self.connection_qubits[-1])
            ctrl(self.connection_qubits[:-1], X)(self.connection_qubits[-1])
            H(self.connection_qubits[-1])
            X(self.connection_qubits)

            ket_state_prep(amplitude, self.connection_qubits)

    def __shift(self, connection : int) -> None:

        binary_connection = bin(connection)[2:].zfill(self.num_qubits_connections)

        binary_node1 = bin(self._connections[connection].node_1)[2:].zfill(self.num_qubits_nodes) #getting little endian representation
        binary_node2 = bin(self._connections[connection].node_2)[2:].zfill(self.num_qubits_nodes) #getting little endian representation

        state = [int(binary_connection[i]) for i in range(len(binary_connection))]

        for qubit in range(self.num_qubits_nodes):
            if binary_node1[qubit] != binary_node2[qubit]:
                ctrl(self.connection_qubits, X, state)(self.node_qubits[qubit])

    def simulate(
        self, steps : int,
        register_probabilities : str = "last", # "all", "last" or "none"
        starting_node : int = -1,
        state_prep_list : list[float] = []
    ):

        if steps < 1:
            raise ValueError("steps must be a positive integer")

        if register_probabilities != "last" and register_probabilities != "all" and register_probabilities != "none":
            raise ValueError("register_probabilities must be: 'all', 'last' or 'none'")

        if state_prep_list != [] and starting_node != -1:
            raise ValueError("starting_node and state_prep_list are mutually exclusive, please choose one")

        # Decided to simply ignore negative numbers, might change it in the future
        elif starting_node >= 0:
            if starting_node > self.num_nodes-1:
                raise ValueError(f"starting_node must be between 0 and {self.num_nodes-1}")
            self.__prepareCircuitFromStartingNode(starting_node)

        elif state_prep_list != []:
            if len(state_prep_list) != 2 ** (self.num_qubits_total):
                raise ValueError(f"state_prep_list must have {2 ** (self.num_qubits_total)} elements, one for each possible binary state")
            ket_state_prep(state_prep_list, self.all_qubits)

        else:
            self.__prepareCircuit()

        self.steps = steps

        for _ in range(steps):

            for node in range(self.num_nodes):
                self.__coin(node)

            for connection in range(self.num_connections):
                self.__shift(connection)


            if register_probabilities == "all":
                self.__getProbabilities()

        if register_probabilities == "last":
            self.__getProbabilities()

    def plotProbabilities(self, print_connection_mapping : bool, projection : str = "2d", nodes : list[int] = []) -> None:

        if self.probabilities == []:
            raise ValueError("No probabilities available")

        if len(self.probabilities) == 1: # In case of only registering the final probability
            fig, ax = plt.subplots()
            ax.bar(range(self.num_nodes), self.probabilities[0])

            ax.set_xlabel("nodes")
            ax.set_xticks(list(range(self.num_nodes)))

            ax.set_ylabel("probability")

            #plt.bar(range(self.num_nodes), self.probabilities[0])

        else:

            nodes.sort()
            nodes = list(set(nodes)) # removing duplicates

            if projection.lower() == "3d":

                x = []
                y = []
                
                if nodes == []:
                    z_size = self.num_nodes * self.steps
                else:
                    z_size = len(nodes) * self.steps
                
                z = [0 for i in range(z_size)]

                dx = [0.5 for i in range(z_size)]
                dy = [0.5 for i in range(z_size)]
                dz = []

                fig, ax = plt.subplots(subplot_kw={"projection" : "3d"})
                for i in range(self.steps):
                    if nodes == []:
                        x += [j - 0.25 for j in range(self.num_nodes)]
                        y += [i + 0.75 for k in range(self.num_nodes)]
                        dz += self.probabilities[i]
                    else:
                        x += [j - 0.25 for j in range(len(nodes))]
                        y += [i + 0.75 for k in nodes]
                        temp = [0 for _ in range(len(nodes))]
                        for j in range(len(nodes)):
                            temp[j] = self.probabilities[i][nodes[j]]
                        dz += temp.copy()

                print(x)

                ax.set_xlabel("nodes")
                if nodes == []:
                    ax.set_xticks(list(range(self.num_nodes)))
                else:
                    ax.set_xticks(list(range(len(nodes))), nodes)

                ax.set_ylabel("steps")
                ax.set_yticks(list(range(1, self.steps + 1)))

                ax.set_zlabel("probability")

                cmap = cm.inferno
                norm = Normalize(vmin=min(dz), vmax=max(dz))
                colors = cmap(norm(dz))

                ax.bar3d(x, y, z, dx, dy, dz, shade=True, color=colors, zsort='max')


            if projection.lower() == "2d":

                if len(nodes) > 10:
                    raise ValueError("This type of plotting only allows 10 nodes visible at once for clarity")

                if nodes == [] and self.num_nodes > 10:
                    raise ValueError("There are more than 10 nodes on this graph, please specify at maximum 10 nodes to view at a time")

                colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k', 'tab:purple', 'tab:orange', 'tab:pink']
                used_colors = 0

                fig, ax = plt.subplots()

                ax.set_xlabel("steps")
                ax.set_xticks(list(range(1, self.steps + 1)))

                ax.set_ylabel("probability")

                x = [i+1 for i in range(self.steps)]
                zeroes = [0 for i in range(self.steps)]

                plots = []

                for i in range(self.num_nodes):
                    y = []

                    for j in range(self.steps):

                        if (nodes != [] and i in nodes) or (nodes == []):
                            y.append(self.probabilities[j][i])

                    if (y != []) and (nodes != [] or y != zeroes):
                        temp, = ax.plot(x, y, c=colors[used_colors], marker='o', linestyle="-.", label="Node " + str(i))
                        used_colors += 1
                        plots.append(temp)

                ax.legend(handles=plots)

        if print_connection_mapping:
            print("\n\t|NODE_1> <---[CONNECTION_ID]---> |NODE_2>\n")
            for i in range(self.num_connections):
                print(f"|{self._connections[i].node_1}> <---[{i} / {bin(i)[2:].zfill(self.num_qubits_connections)}]---> |{self._connections[i].node_2}>")

        plt.show()

    def draw(self, show_labels : bool = True) -> None:
        nx.draw(self.networkx_graph, with_labels=show_labels, node_size=400, font_size=13, node_color="black", font_color="white")
        plt.show()

    def reset(self) -> None:

        self.probabilities : list[list[float]] = []
        self.steps = 0

        self.proc = Process()
        self.all_qubits = self.proc.alloc(self.num_qubits_total)
        self.node_qubits = self.all_qubits[: self.num_qubits_nodes]
        self.connection_qubits = self.all_qubits[self.num_qubits_nodes :]

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

    G = nx.gnp_random_graph(100, 0.3)
    print("Nodes:", G.number_of_nodes(), "Edges:", G.number_of_edges())
    print("Registrando probabilidades a cada passo")
    test_DiscreteWalk = DTQW(G)
    test_DiscreteWalk.simulate(steps=1, register_probabilities='all')

    #test_DiscreteWalk.simulate(steps=10, register_probabilities='all', starting_node=0)
    #test_DiscreteWalk.plotProbabilities(False, projection="2d", nodes=[0])