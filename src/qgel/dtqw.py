#import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
from ket import *
from ket.qulib.prepare import state as ket_state_prep
from matplotlib import cm
from matplotlib.colors import Normalize
from networkx.classes.graph import Graph
from numpy import array, ceil, complexfloating, dtype, log2, ndarray, sqrt
from numpy.linalg import norm
from typeguard import TypeCheckError, check_type


@dataclass
class Node:
    """
    Simple dataclass used for packaging data of neighbors nodes
    and the weights of the connections between those nodes.

    Examples
    --------
    In the graph described by this weighted adjacency matrix::
            [0,    1, 0.5]
            [1,    0,  -3]
            [0.5, -3,   0]

    The neighbors of node 0 are 1 and 2, because they're connected
    and the weights are, respectively, 1 and 0.5.

    Therefore, we can make build the nodes such as:
    >>> n0 = Node(neighbors=[1, 2], weights=[1,  0.5])
    >>> n1 = Node(neighbors=[0, 2], weights=[1,   -3])
    >>> n2 = Node(neighbors=[0, 1], weights=[0.5, -3])
    """

    neighbors: list[int]
    weights: list[complex]

class DTQW:
    """
    Class for discrete-time quantum walks.

    A DTQW stores a list of Nodes, the respective nx Graph and
    the circuit if it has been simulated.

    Parameters
    ----------
    graph : Graph | list[list[int]] | list[Node] | ndarray[Any, dtype[complexfloating]]
        Representation of the graph you want to work on.

    Examples
    --------
    Creating dtqw from a simple non-weighted adjacency matrix:
    >>> adj_matrix = [
    >>>     [0, 1, 1],
    >>>     [1, 0, 0],
    >>>     [0, 1, 0]
    >>> ]
    >>> walk = DTQW(graph=adj_matrix)

    From a nx Graph:
    >>> G = nx.gnp_random_graph(100, 0.3) # Random graph with 100 nodes
    >>> walk = DTQW(G)

    **Methods:**

    Simulate and build circuit:

    >>> walk.simulate(steps=10) # Circuit for dtqw of 10 steps

    Simulating with more control:

    >>> walk.simulate(steps=12, register_probabilities="all", starting_node=2)
    >>> walk.simulate(steps=7, register_probabilities="none", state_prep_list=[1/sqrt(2),0,1/sqrt(2),0])

    Plotting the probability (ampitude) of each node in a plt graph:

    >>> walk.plot_probabilities()
    >>> walk.plot_probabilities(projection="3d")
    >>> walk.plot_probabilities(projection="2d", nodes=[2, 5, 9])

    Drawing the graph with plt:

    >>> walk.draw()
    >>> walk.draw(show_labels=False)

    Getting the degree of a specific node:

    >>> deg: dict[str, complex] = walk.get_degree(node_id=3)

    Resetting the dtqw for another simulation:

    >>> walk.reset()

    Raises
    ------
    ValueError
        When inputting a type not supported as a graph representation.
    """

    def __handle_nxgraph(self, graph) -> bool:

        try:
            check_type(graph, Graph)

            self._nxgraph: Graph = graph

            self._adjacency_list = [Node([],[]) for _ in graph.nodes]
            self._num_nodes = len(self._adjacency_list)

            for node, adj in graph.adjacency():

                self._adjacency_list[node] = Node(
                    list(adj.keys()), 
                    [w["weight"] if w != {} else 1.0 for w in adj.values()]
                )

        except TypeCheckError: return False

        else: return True

    def __handle_list_of_list(self, graph) -> bool:

        try:
            check_type(graph, list[list[int]])

            G: Graph = nx.from_numpy_array(array(graph))
            self.__handle_nxgraph(G)

        except TypeCheckError: return False

        else: return True

    def __handle_list_of_node(self, graph) -> bool:

        try:
            check_type(graph, list[Node])

            self._adjacency_list: list[Node] = graph
            self._num_nodes = len(self._adjacency_list)

            self._nxgraph = nx.Graph()
            self._nxgraph.add_nodes_from([node_id for node_id in range(self._num_nodes)])

            for node_id in range(self._num_nodes):
                node = self._adjacency_list[node_id]
                edges = [(node_id, node.neighbors[i], node.weights[i]) for i in range(len(node.neighbors))]
                self._nxgraph.add_weighted_edges_from(edges)

        except TypeCheckError: return False

        else: return True

    def __handle_ndarray(self, graph) -> bool:

        try:
            check_type(graph, ndarray[Any, dtype[complexfloating]])

            G: Graph = nx.from_numpy_array(graph)
            self.__handle_nxgraph(G)

        except TypeCheckError: return False

        else: return True

    def __handle_input(self, graph) -> None:

        if not (self.__handle_nxgraph(graph)
            or self.__handle_list_of_list(graph)
            or self.__handle_list_of_node(graph)
            or self.__handle_ndarray(graph)
        ):
            raise TypeError("Type of graph not supported")



    def __init__( self, graph: Graph | list[list[int]] | list[Node] | ndarray[Any, dtype[complexfloating]] ):

        self._adjacency_list: list[Node] = []
        self._num_nodes: int
        self._nxgraph : Graph

        self.__handle_input(graph)

        if self._num_nodes < 2:
            raise ValueError("Number of nodes must be greater than 1")

        self._num_qubits_node = int(ceil(log2(self._num_nodes)))
        self._num_qubits_total: int = 2 * self._num_qubits_node

        self._proc = Process()
        self._all_qubits: Quant = self._proc.alloc(self._num_qubits_total)
        self._first_node_qubits: list[Quant] = self._all_qubits[:self._num_qubits_node]
        self._second_node_qubits: list[Quant] = self._all_qubits[self._num_qubits_node:]

        self._probabilities : list[list[float]] = []
        self._steps : int

        self.__num_binary_states_node: int = 2 ** self._num_qubits_node
        self.__num_binary_states_total: int = 2 ** self._num_qubits_total



    def __get_amplitude(self, node_id: int) -> list[complex]:

        node = self._adjacency_list[node_id]

        length = norm(node.weights)

        if length == 0.0:
            raise ValueError("Norm of node must be non-zero")

        normalized_weights = node.weights.copy() / length  # ty: ignore[unsupported-operator] | Python supports complex by float division

        amplitude: list[complex] = [0.0 for _ in range(self.__num_binary_states_node)]

        for i in range(len(normalized_weights)):
            amplitude[node.neighbors[i]] = normalized_weights[i]

        return amplitude

    def __get_state(self, node_id: int) -> list[int]:
        binary = bin(node_id)[2:].zfill(self._num_qubits_node)
        return [int(bit) for bit in binary]



    def __prepare_circuit_from_node(self, starting_node_id: int) -> None:

        # Making the amplitude of the node given 100% and all others 0%
        amplitudes: list[complex] = [0.0 for _ in range(self.__num_binary_states_node)]
        amplitudes[starting_node_id] = 1.0

        ket_state_prep(amplitudes, self._first_node_qubits)  # ty: ignore[invalid-argument-type] | Ket handles list[Quant]

        # Adjusting neighbors amplitudes accordingly
        amplitude_node = self.__get_amplitude(starting_node_id)
        ket_state_prep(amplitude_node, self._second_node_qubits)  # ty: ignore[invalid-argument-type] | Ket handles list[Quant]

    def __prepare_circuit(self) -> None:

        # Uniformly distributing the amplitude across all nodes
        amplitudes: list[complex] = [1/sqrt(self._num_nodes) if i < self._num_nodes else 0.0 for i in range(self.__num_binary_states_node)]
        ket_state_prep(amplitudes, self._first_node_qubits)            # ty: ignore[invalid-argument-type] | Ket handles list[Quant]

        for node_id, node in enumerate(self._adjacency_list):

            temp_amplitude = self.__get_amplitude(node_id)
            amps: list[complex] = [1/sqrt(len(node.weights)) if a != 0.0 else 0.0 for a in temp_amplitude]
            state = self.__get_state(node_id)

            with control(self._first_node_qubits, state):   # ty: ignore[invalid-argument-type] | Ket handles list[Quant]
                ket_state_prep(amps, self._second_node_qubits)         # ty: ignore[invalid-argument-type] | Ket handles list[Quant]

    def __coin(self, node_id: int) -> None:

        amplitude = self.__get_amplitude(node_id)
        state = self.__get_state(node_id)

        with control(self._first_node_qubits, state):                                   # ty: ignore[invalid-argument-type] | Ket handles list[Quant]

            adj(ket_state_prep)(amplitude, self._second_node_qubits)                              # ty: ignore[invalid-argument-type, too-many-positional-arguments] | Ket being quirky

            X(self._second_node_qubits)                                                                # ty: ignore[invalid-argument-type] | Ket handles list[Quant]
            H(self._second_node_qubits[-1])
            ctrl(self._second_node_qubits[:-1], X)(self._second_node_qubits[-1])   # ty: ignore[invalid-argument-type] | Ket handles list[Quant]
            H(self._second_node_qubits[-1])
            X(self._second_node_qubits)                                                                # ty: ignore[invalid-argument-type] | Ket handles list[Quant]

            ket_state_prep(amplitude, self._second_node_qubits)                                    # ty: ignore[invalid-argument-type] | Ket handles list[Quant]

    def __shift(self) -> None:
        for i in range(self._num_qubits_node):
            SWAP(self._first_node_qubits[i], self._second_node_qubits[i])



    def __get_probabilities(self) -> None:

        dump_dict = dump(self._all_qubits).get()                                   # ty: ignore[invalid-argument-type] | Ket handles Quant

        for i in dump_dict:
            dump_dict[i] = dump_dict[i] ** 2

        data: list[float] = [0.0 for _ in range(self._num_nodes)]

        for key in dump_dict:
            binary_total = bin(key)[2:].zfill(self._num_qubits_total)
            binary_node = binary_total[:self._num_qubits_node]
            data[int(binary_node, 2)] += abs(dump_dict[key])

        self._probabilities.append(data)



    def simulate(
        self,
        steps : int,
        register_probabilities : str = "last", # "all", "last" or "none"
        starting_node_id : int = -1,
        state_prep_list : list[complex] = []  # noqa: B006
    ) -> None:

        """
        Simulate the DTQW and builds the circuit.

        Parameters
        ----------
        steps : int
            Amount of steps the DTQW will take.

        register_probabilities: str
            Which probabilities will the class store, if any. Options: 'all', 'last' or 'none'. If steps == 1, 'all' and 'last' make no difference.
            (optional, defaults to 'last').

        starting_node_id : int
            Defines the starting point of the DTQW, if you choose to start from a single point. -1 means "start from all nodes".
            (optional, defaults to -1).
            #### !INCOMPATIBLE WITH state_prep_list

        state_prep_list: list[complex]
            List of amplitudes for the initial state of the DTQW. empty means uniform distribution among all possible states.
            (optional, defaults to empty).
            #### !INCOMPATIBLE WITH starting_node_id

        Raises
        ------
        ValueError
            When steps is a non-positive value.

        ValueError
            When register_probabilities is not 'all', 'last' or 'none'.

        ValueError
            When register_probabilities and starting_node_id are both defined as a non-default value.

        ValueError
            When starting_node_id is outside the range between 0 and self._num_node-1.

        ValueError
            When state_prep_list element count is not 2 ** (self._num_qubits_total).
        """

        if steps < 1:
            raise ValueError("steps must be a positive integer")

        if register_probabilities != "last" and register_probabilities != "all" and register_probabilities != "none":
            raise ValueError("register_probabilities must be: 'all', 'last' or 'none'")

        if state_prep_list != [] and starting_node_id != -1:
            raise ValueError("starting_node_id and state_prep_list are mutually exclusive, please choose one")

        # Decided to simply ignore negative numbers, might change it in the future
        elif starting_node_id >= 0:
            if starting_node_id > self._num_nodes-1:
                raise ValueError(f"starting_node_id must be between 0 and {self._num_nodes-1}")
            self.__prepare_circuit_from_node(starting_node_id)

        elif state_prep_list != []:
            if len(state_prep_list) != 2 ** (self._num_qubits_total):
                raise ValueError(f"state_prep_list must have {2 ** (self._num_qubits_total)} elements, one for each possible binary state")
            ket_state_prep(state_prep_list, self._all_qubits)

        else:
            self.__prepare_circuit()

        self._steps = steps

        for _ in range(steps):

            for node_id in range(len(self._adjacency_list)):
                self.__coin(node_id)

            self.__shift()

            if register_probabilities == "all":
                self.__get_probabilities()

        if register_probabilities == "last":
            self.__get_probabilities()


    def plot_probabilities(
        self,
        projection : str = "2d",
        nodes : list[int] = []  # noqa: B006
    ) -> None:

        """
        Plots the amplitude(s) of each node using plt

        Parameters
        ----------
        projection : str
            Projection of the graph, '2d' or '3d'. If it has only one probability list,
            even when 3d is set, the graph will be '2d'. (optional, defaults to '2d')

        nodes: list[int]
            List of the node ids you want to see, only those nodes probabilities will be shown. Empty means all nodes.
            (optional, defaults to empty)

        Raises
        ------
        ValueError
            When trying to plot with no probability list stored. Simulate the DTQW first.

        ValueError
            When trying to plot multiple amplitudes with the '2d' projection and more than 10 nodes at once.
            in this case, only the '3d' projection will allow visualization or choose at most 10 nodes to view.
        """

        if self._probabilities == []:
            raise ValueError("No probabilities available")

        if len(self._probabilities) == 1: # In case of only registering the final probability
            _, ax = plt.subplots()
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

                _, ax = plt.subplots(subplot_kw={"projection" : "3d"})
                for i in range(self._steps):
                    if nodes == []:
                        x += [j - 0.25 for j in range(self._num_nodes)]
                        y += [i + 0.75 for k in range(self._num_nodes)]
                        dz += self._probabilities[i]
                    else:
                        x += [j - 0.25 for j in range(len(nodes))]
                        y += [i + 0.75 for k in nodes]
                        temp = [0.0 for _ in range(len(nodes))]
                        for j in range(len(nodes)):
                            temp[j] = self._probabilities[i][nodes[j]]
                        dz += temp.copy()

                ax.set_xlabel("nodes")
                if nodes == []:
                    ax.set_xticks(list(range(self._num_nodes)))
                else:
                    ax.set_xticks(list(range(len(nodes))), nodes)        # ty: ignore[invalid-argument-type] | plt handles list[int] 

                ax.set_ylabel("steps")
                ax.set_yticks(list(range(1, self._steps + 1)))

                ax.set_zlabel("probability")                                          # ty: ignore[unresolved-attribute] | plt being quirky

                cmap = cm.inferno
                norm = Normalize(vmin=min(dz), vmax=max(dz))
                colors = cmap(norm(dz))

                ax.bar3d(x, y, z, dx, dy, dz, shade=True, color=colors, zsort='max')  # ty: ignore[unresolved-attribute] | plt being quirky


            if projection.lower() == "2d":

                if len(nodes) > 10 or self._num_nodes > 10:
                    raise ValueError("You are trying to visualize more than 10 nodes at once, please specify at maximum 10 nodes to view at a time. This constraint is here to make the plotting more cohesive")

                colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k', 'tab:purple', 'tab:orange', 'tab:pink']
                used_colors = 0

                _, ax = plt.subplots()

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
        plt.clf()

    def draw(self, show_labels : bool = True) -> None:
        """
        Draws the graph with nx.draw and plt.

        Parameters
        ----------
        show_labels : bool
            defines if the drawing will show the node's id (optional, defaults to True).
        """

        nx.draw(self._nxgraph, with_labels=show_labels, node_size=400, font_size=13, node_color="black", font_color="white")
        plt.show()
        plt.clf()

    def reset(self) -> None:
        """
        Resets the DTQW list of probabilities like it has never been simulated.
        """

        self._probabilities : list[list[float]] = []
        self._steps = 0

        self._proc = Process()
        self._all_qubits = self._proc.alloc(self._num_qubits_total)
        self._first_node_qubits = self._all_qubits[:self._num_qubits_node]
        self._second_node_qubits = self._all_qubits[self._num_qubits_node:]

    def get_degree(self, node_id: int) -> dict[str, complex]:
        """
        Given a node_id, returns the degree of that node.

        Parameters
        ----------
        node_id : int
            number ranging from 0 to self._num_nodes-1.

        Returns
        -------
        dict[str, complex]
            dictionary with "in" and "out" entries, each with its corresponding value.

        Raises
        ------
        ValueError
            When inputting a value outside the range between 0 and self._num_node-1.
        """

        if node_id > self._num_nodes-1 or node_id < 0:
            raise ValueError(f"node_id must be between 0 and {self._num_nodes-1}")

        degree: dict[str, complex] = {}
        degree["in"] = 0.0
        degree["out"] = 0.0

        for n_id, node in enumerate(self._adjacency_list):

            if n_id == node_id:
                degree["out"] = sum(node.weights)

            elif n_id in node.neighbors:
                index = node.neighbors.index(n_id)
                degree["in"] += node.weights[index]

        return degree


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

    d = DTQW(study_matrix)
    d.simulate(1)
    nx.gnp_random_graph(100, 0.3)
    #d.plot_probabilities()
    d.draw()