from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library.standard_gates import XGate
import numpy as np
import matplotlib.pyplot as plt

from qiskit.quantum_info import Statevector

# ===================================
# Global variables
# ===================================

num_nodes = 8
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

#num_nodes = 4
#adj_matrix = [
#    [0, 1, 1, 1],
#    [1, 0, 1, 0],
#    [1, 1, 0, 0],
#    [1, 0, 0, 0]
#]

#num_nodes = 5
#adj_matrix = [
#    [0, 1, 0, 1, 0],
#    [1, 0, 1, 0, 0],
#    [0, 1, 0, 0, 1],
#    [1, 0, 0, 0, 0],
#    [0, 1, 0, 0, 0]
#]

nodes : dict = {}
connections : dict = {}

num_connections = 0

steps : int

log_num_nodes : int
log_num_connections : int
total_num_qubits : int

CXGate : XGate

application_list_CXGate : list
application_list_coin : list
application_list_shift : list

gates : list = [] # Stored gates so when running big circuits it avoids wasteful repeating of gate calculations

qc : QuantumCircuit

def gatherData() -> None:

    # Telling python to update the global variables instead of creating local ones
    global nodes
    global connections

    global num_connections
    
    global steps
    
    global log_num_nodes
    global log_num_connections
    global total_num_qubits

    global CXGate

    global application_list_CXGate
    global application_list_coin
    global application_list_shift

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

    steps = 1#int(input("How many steps should the walk simulate?: "))

    log_num_nodes = int(np.ceil(np.log2(num_nodes)))

    log_num_connections = int(np.ceil(np.log2(num_connections)))

    total_num_qubits = log_num_nodes + log_num_connections

    CXGate = XGate().control(log_num_connections-1)
    application_list_CXGate = list(range(log_num_connections))

    application_list_coin = list(range(log_num_connections, total_num_qubits)) + list(range(log_num_connections))

    application_list_shift = list(range(total_num_qubits))

def prepareCircuit() -> None:

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

def getAmplitudeOfNode(node : int) -> list:

    amplitudes = []

    inverse_of_sqrt_degree = 1/np.sqrt(nodes[node]["degree"])

    for i in range(2 ** log_num_connections):
        amplitudes.append(0)

    for connection in nodes[node]["connections"]:
        amplitudes[connection] = inverse_of_sqrt_degree

    return amplitudes

def addCoin(node: int) -> None:

    binary_node = bin(node)[2:].zfill(log_num_nodes)

    sub_qc_coin = QuantumCircuit(log_num_connections)

    amplitudes = getAmplitudeOfNode(node)

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

    gates.append([coin_gate, application_list_coin])

    qc.append(coin_gate, application_list_coin)

def addShift(connection : int) -> None:

    binary_connection = bin(connection)[2:].zfill(log_num_connections)

    sub_qc_shift = QuantumCircuit(log_num_nodes)

    binary_node1 = bin(connections[connection]["node1"])[2:].zfill(log_num_nodes)[::-1] #getting little endian representation
    binary_node2 = bin(connections[connection]["node2"])[2:].zfill(log_num_nodes)[::-1] #getting little endian representation

    for qubit in range(log_num_nodes):
        if binary_node1[qubit] != binary_node2[qubit]:
            sub_qc_shift.x(qubit)

    shift_gate = sub_qc_shift.to_gate().control(log_num_connections, label=f"S{connection}", ctrl_state=binary_connection)

    gates.append([shift_gate, application_list_shift])
    qc.append(shift_gate, application_list_shift)

def getProbabilities() -> None:

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

if __name__ == "__main__":

    gatherData()

    qr_nodes = QuantumRegister(log_num_nodes, 'q')
    qr_connections = QuantumRegister(log_num_connections, 'l')
    cr = ClassicalRegister(log_num_nodes, 'c')

    qc = QuantumCircuit(qr_connections, qr_nodes, cr)

    prepareCircuit()

    for node in range(num_nodes):
        addCoin(node)

    for connection in range(num_connections):
        addShift(connection)

    for step in range(1, steps):
        for gate in range(len(gates)):
            qc.append(gates[gate][0], gates[gate][1]) # Appending the gate (gate[gate][0]) to it's application list (gates[gate][1])

    getProbabilities()

    qc.draw(output="mpl",reverse_bits=True) 
    plt.show()
