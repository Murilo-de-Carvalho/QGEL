from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library.standard_gates import XGate
import numpy as np
import matplotlib.pyplot as plt

from qiskit.quantum_info import Statevector

def __prepareCircuit(qc : QuantumCircuit, total_num_qubits : int, nodes : dict[any], log_num_nodes : int, log_num_connections : int, num_nodes : int) -> None:

    amplitudes = []

    num_possibilities = 2 ** (total_num_qubits)

    for i in range(num_possibilities):
        amplitudes.append(0)

    for node in range(num_nodes):

        inverse_of_sqrt_degree = 1/np.sqrt(nodes[node]["degree"])

        for connection in nodes[node]["connections"]:
            binary_node = bin(node)[2:].zfill(log_num_nodes)
            binary_connection = bin(connection)[2:].zfill(log_num_connections)
            state = binary_node + binary_connection
            amplitudes[int(state, 2)] = inverse_of_sqrt_degree

    inverse_of_sqrt_num_nodes = 1/np.sqrt(num_nodes)

    for i in range(num_possibilities):
        if amplitudes[i] != 0: # mathematically doesn't make a difference but is prittier to print and skip some divisions and multiplications
            amplitudes[i] *= inverse_of_sqrt_num_nodes

    qc.prepare_state(amplitudes)

def __getAmplitudeOfNode(node : int, nodes : dict[any], log_num_connections : int) -> list[float]:

    amplitudes = []

    inverse_of_sqrt_degree = 1/np.sqrt(nodes[node]["degree"])

    for i in range(2 ** log_num_connections):
        amplitudes.append(0)

    for connection in nodes[node]["connections"]:
        amplitudes[connection] = inverse_of_sqrt_degree

    return amplitudes

def __addCoin(node: int, nodes : dict[any], application_list : list[int], gates : list[any], log_num_nodes : int, log_num_connections : int, CXGate : any, application_list_CXGate : list[int]) -> None:

    binary_node = bin(node)[2:].zfill(log_num_nodes)

    sub_qc_coin = QuantumCircuit(log_num_connections)

    amplitudes = __getAmplitudeOfNode(node, nodes, log_num_connections)

    # Apply Si dagger
    sub_qc_coin.prepare_state(amplitudes).inverse()

    for i in range(0, log_num_connections):
        sub_qc_coin.x(i)

    sub_qc_coin.h(0)
    sub_qc_coin.append(CXGate, application_list_CXGate[::-1])
    sub_qc_coin.h(0)

    for i in range(0, log_num_connections):
        sub_qc_coin.x(i)

    # Apply Si
    sub_qc_coin.prepare_state(amplitudes)

    coin_gate = sub_qc_coin.to_gate().control(log_num_nodes, label=f"C{node}", ctrl_state=binary_node)

    gates.append([coin_gate, application_list])

def __addShift(connection : int, connections : dict[str, int], application_list : list[int], gates : list[any], log_num_nodes : int, log_num_connections : int) -> None:

    binary_connection = bin(connection)[2:].zfill(log_num_connections)

    sub_qc_shift = QuantumCircuit(log_num_nodes)

    binary_node1 = bin(connections[connection]["node1"])[2:].zfill(log_num_nodes)[::-1] #getting little endian representation
    binary_node2 = bin(connections[connection]["node2"])[2:].zfill(log_num_nodes)[::-1] #getting little endian representation

    for qubit in range(log_num_nodes):
        if binary_node1[qubit] != binary_node2[qubit]:
            sub_qc_shift.x(qubit)

    shift_gate = sub_qc_shift.to_gate().control(log_num_connections, label=f"S{connection}", ctrl_state=binary_connection)

    gates.append([shift_gate, application_list])

def __getProbabilities(qc : QuantumCircuit, num_connections : int, log_num_nodes : int, log_num_connections : int, num_nodes : int) -> None:

    sv = Statevector(qc)
    prob_dict = sv.probabilities_dict()

    probabilities = []

    for node in range(num_nodes):

        prob = 0

        for connection in range(num_connections):

            binary_node = bin(node)[2:].zfill(log_num_nodes)
            binary_connection = bin(connection)[2:].zfill(log_num_connections)

            full_binary = binary_node + binary_connection

            prob += prob_dict[full_binary] if full_binary in prob_dict else 0

        probabilities.append(prob)

    plt.bar(range(num_nodes), probabilities)
    plt.show()

def discreteTimeWalk(adj_matrix : list[list[int]], steps : int, show_probabilities : bool, print_connection_mapping : bool) -> QuantumCircuit:

    num_nodes = len(adj_matrix)

    nodes = {}

    connections : dict[str, int] = {}

    num_connections = 0

    for i in range(num_nodes):
        nodes[i] = {"degree": 0, "connections": []}

    for i in range(num_nodes):
        for j in range(i, num_nodes):
            if (adj_matrix[i][j]):
                connections[num_connections] = {"node1": i, "node2": j}
                nodes[i]["degree"] += 1
                nodes[i]["connections"].append(num_connections)
                nodes[j]["degree"] += 1
                nodes[j]["connections"].append(num_connections)
                num_connections += 1

    gates : list[list[int | list[int]]] = []

    log_num_nodes = int(np.ceil(np.log2(num_nodes)))

    log_num_connections = int(np.ceil(np.log2(num_connections)))

    total_num_qubits = log_num_nodes + log_num_connections

    CXGate = XGate().control(log_num_connections-1)
    application_list_CXGate = list(range(log_num_connections))

    application_list_coin = list(range(log_num_connections, total_num_qubits)) + list(range(log_num_connections))

    application_list_shift = list(range(total_num_qubits))

    qr_nodes = QuantumRegister(log_num_nodes, 'q')
    qr_connections = QuantumRegister(log_num_connections, 'l')
    cr = ClassicalRegister(log_num_nodes, 'c')

    qc = QuantumCircuit(qr_connections, qr_nodes, cr)

    __prepareCircuit(qc, total_num_qubits, nodes, log_num_nodes, log_num_connections, num_nodes)

    for node in range(num_nodes):
        __addCoin(node, nodes, application_list_coin, gates, log_num_nodes, log_num_connections, CXGate, application_list_CXGate)

    for connection in range(num_connections):
        __addShift(connection, connections, application_list_shift, gates, log_num_nodes, log_num_connections)

    for _ in range(0, steps):
        for gate in range(len(gates)):
            qc.append(gates[gate][0], gates[gate][1]) # Appending the gate (gate[gate][0]) to it's application list (gates[gate][1])

    if show_probabilities:
        __getProbabilities(qc, num_connections, log_num_nodes, log_num_connections, num_nodes)

    if print_connection_mapping:
        print("\n\t|NODE_1> <---[CONNECTION_ID]---> |NODE_2>\n")
        for i in range(num_connections):
            print(f"|{connections[i]["node1"]}> <---[{i} / {bin(i)[2:].zfill(log_num_connections)}]---> |{connections[i]["node2"]}>")
            
    return qc

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

    test = discreteTimeWalk(adj_matrix, 2, True, True)