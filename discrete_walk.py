from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library.standard_gates import XGate
import numpy as np
import matplotlib.pyplot as plt

from qiskit.quantum_info import Statevector

class DiscreteTimeWalk:

    def __init__(self, adj_matrix : list[list[int]]):

        self.adj_matrix = adj_matrix
        self.num_nodes = len(adj_matrix)
        self.__nodes = {}
        self.__connections : dict[str, int] = {}
        self.num_connections = 0

        for i in range(self.num_nodes):
            self.__nodes[i] = {"degree": 0, "connections": []}

        for i in range(self.num_nodes):
            for j in range(i, self.num_nodes):
                if (adj_matrix[i][j]):
                    self.__connections[self.num_connections] = {"node1": i, "node2": j}
                    self.__nodes[i]["degree"] += 1
                    self.__nodes[i]["connections"].append(self.num_connections)
                    self.__nodes[j]["degree"] += 1
                    self.__nodes[j]["connections"].append(self.num_connections)
                    self.num_connections += 1

        self.gates : list[list[int | list[int]]] = []

        self.log_num_nodes = int(np.ceil(np.log2(self.num_nodes)))

        self.log_num_connections = int(np.ceil(np.log2(self.num_connections)))

        self.total_num_qubits = self.log_num_nodes + self.log_num_connections

        self.__CXGate = XGate().control(self.log_num_connections-1)
        self.__application_list_CXGate = list(range(self.log_num_connections))

        self.__application_list_coin = list(range(self.log_num_connections, self.total_num_qubits)) + list(range(self.log_num_connections))

        self.__application_list_shift = list(range(self.total_num_qubits))

        qr_nodes = QuantumRegister(self.log_num_nodes, 'q')
        qr_connections = QuantumRegister(self.log_num_connections, 'l')
        cr = ClassicalRegister(self.log_num_nodes, 'c')

        self.probabilities : list[list[float]] = []
        self.qc = QuantumCircuit(qr_connections, qr_nodes, cr)



    def __prepareCircuit(self) -> None:

        amplitudes = []

        num_possibilities = 2 ** (self.total_num_qubits)

        for i in range(num_possibilities):
            amplitudes.append(0)

        for node in range(self.num_nodes):

            inverse_of_sqrt_degree = 1/np.sqrt(self.__nodes[node]["degree"])

            for connection in self.__nodes[node]["connections"]:
                binary_node = bin(node)[2:].zfill(self.log_num_nodes)
                binary_connection = bin(connection)[2:].zfill(self.log_num_connections)
                state = binary_node + binary_connection
                amplitudes[int(state, 2)] = inverse_of_sqrt_degree

        inverse_of_sqrt_num_nodes = 1/np.sqrt(self.num_nodes)

        for i in range(num_possibilities):
            if amplitudes[i] != 0: # mathematically doesn't make a difference but is prittier to print and skip some divisions and multiplications
                amplitudes[i] *= inverse_of_sqrt_num_nodes

        self.qc.prepare_state(amplitudes)



    def __getAmplitudeOfNode(self, node : int) -> list[float]:

        amplitudes = []

        inverse_of_sqrt_degree = 1/np.sqrt(self.__nodes[node]["degree"])

        for i in range(2 ** self.log_num_connections):
            amplitudes.append(0)

        for connection in self.__nodes[node]["connections"]:
            amplitudes[connection] = inverse_of_sqrt_degree

        return amplitudes



    def __addCoin(self, node : int) -> None:

        binary_node = bin(node)[2:].zfill(self.log_num_nodes)

        sub_qc_coin = QuantumCircuit(self.log_num_connections)

        amplitudes = self.__getAmplitudeOfNode(node)

        # Apply Si dagger
        sub_qc_coin.prepare_state(amplitudes).inverse()

        for i in range(0, self.log_num_connections):
            sub_qc_coin.x(i)

        sub_qc_coin.h(0)
        sub_qc_coin.append(self.__CXGate, self.__application_list_CXGate[::-1])
        sub_qc_coin.h(0)

        for i in range(0, self.log_num_connections):
            sub_qc_coin.x(i)

        # Apply Si
        sub_qc_coin.prepare_state(amplitudes)

        coin_gate = sub_qc_coin.to_gate().control(self.log_num_nodes, label=f"C{node}", ctrl_state=binary_node)

        self.gates.append([coin_gate, self.__application_list_coin])



    def __addShift(self, connection : int) -> None:

        binary_connection = bin(connection)[2:].zfill(self.log_num_connections)

        sub_qc_shift = QuantumCircuit(self.log_num_nodes)

        binary_node1 = bin(self.__connections[connection]["node1"])[2:].zfill(self.log_num_nodes)[::-1] #getting little endian representation
        binary_node2 = bin(self.__connections[connection]["node2"])[2:].zfill(self.log_num_nodes)[::-1] #getting little endian representation

        for qubit in range(self.log_num_nodes):
            if binary_node1[qubit] != binary_node2[qubit]:
                sub_qc_shift.x(qubit)

        shift_gate = sub_qc_shift.to_gate().control(self.log_num_connections, label=f"S{connection}", ctrl_state=binary_connection)

        self.gates.append([shift_gate, self.__application_list_shift])



    def __getProbabilities(self) -> None:

        sv = Statevector(self.qc)
        prob_dict = sv.probabilities_dict()

        probability_list : list[float] = []

        for node in range(self.num_nodes):

            prob = 0

            for connection in range(self.num_connections):

                binary_node = bin(node)[2:].zfill(self.log_num_nodes)
                binary_connection = bin(connection)[2:].zfill(self.log_num_connections)

                full_binary = binary_node + binary_connection

                prob += prob_dict[full_binary] if full_binary in prob_dict else 0

            probability_list.append(prob)

        self.probabilities.append(probability_list)



    def simulate(self, steps : int, register_all_probabilities : bool):

        self.__prepareCircuit()

        for node in range(self.num_nodes):
            self.__addCoin(node)

        for connection in range(self.num_connections):
            self.__addShift(connection)

        for _ in range(0, steps):
            for gate in range(len(self.gates)):
                self.qc.append(self.gates[gate][0], self.gates[gate][1]) # Appending the gate (gates[gate][0]) to it's application list (gates[gate][1])

            if register_all_probabilities:
                self.__getProbabilities()

        if not register_all_probabilities:
            self.__getProbabilities()



    def plotProbabilities(self, print_connection_mapping : bool) -> None:

        if len(self.probabilities) == 1: # In case of only registering the final probability
            plt.bar(range(self.num_nodes), self.probabilities[0])

        else:
            pass # TODO Find a way to plot in 3D

        if print_connection_mapping:
            print("\n\t|NODE_1> <---[CONNECTION_ID]---> |NODE_2>\n")
            for i in range(self.num_connections):
                print(f"|{self.__connections[i]["node1"]}> <---[{i} / {bin(i)[2:].zfill(self.log_num_connections)}]---> |{self.__connections[i]["node2"]}>")

        plt.show()

if __name__ == "__main__":

    adj_matrix = [
    [0, 0, 1, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [1, 0, 0, 0, 1, 0, 1, 0],
    [0, 0, 0, 0, 1, 1, 0, 0],
    [0, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 0]
    ]

    #adj_matrix = [
    #    [0, 1, 1, 1],
    #    [1, 0, 1, 0],
    #    [1, 1, 0, 0],
    #    [1, 0, 0, 0]
    #]

    #adj_matrix = [
    #    [0, 1, 0, 1, 0],
    #    [1, 0, 1, 0, 0],
    #    [0, 1, 0, 0, 1],
    #    [1, 0, 0, 0, 0],
    #    [0, 1, 0, 0, 0]
    #]

    test = DiscreteTimeWalk(adj_matrix)
    test.simulate(1, False)
    test.plotProbabilities(True)