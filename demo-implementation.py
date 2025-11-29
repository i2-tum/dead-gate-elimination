def gen_random_list_circuit_alike(num_qubits, num_gates, p_single, p_swap):
    # num_qubits: number of qubits.
    # num_gates: number of gates.
    # p_single: probability of generating a single qubit gate.
    # p_swap: probability of generating a SWAP gate.
    
    # Format of gate:
    # [0, q] : single-qubit gate, acting on qubit q.
    # [n, q1,..., qn] : multi-qubit gate, acting on qubit q1, q2, ..., qn.
    # [2, q1, q2] : CNOT gate, q1 is controlling qubit.
    # [-1, i, j] : SWAP gate, acting on qubit i and j.
    res = []
    for _ in range(num_gates):
        if (random.random() < p_single):
            i = random.randint(0, num_qubits - 1)
            res.append([0, i])
        elif (random.random() < p_swap):
            i = random.randint(0, num_qubits - 1)
            j = random.randint(0, num_qubits - 1)
            if i == j:
                j = (j + random.randint(1, num_qubits - 1))%num_qubits
            res.append([-1, i, j])
        else:
            i = random.randint(0, num_qubits - 1)
            j = random.randint(0, num_qubits - 1)
            res.append([2, i, j])
            
    return res

def remove_dead_gates_from_pseudo_list(circuit, invalid_qubits):
    # circuit: input list pretending to be circuit.
    # invalid_qubits: the list of qubits assumed to be invalid.
    circuit.reverse()
    new_circuit = []
    for gate in circuit:
        if gate[0] == 0 and gate[1] in invalid_qubits:
            continue
        if gate[0] == -1 and gate[1] in invalid_qubits:
            invalid_qubits.remove(gate[1])
            invalid_qubits.append(gate[2])
            continue
        if gate[0] == 2 and gate[2] in invalid_qubits:
            continue
        new_circuit.append(gate)
        if gate[0] == 2 and gate[1] in invalid_qubits:
            invalid_qubits.remove(gate[1])
    return new_circuit

def num_removed_dead_gates_of_a_random_circuit(num_qubits, num_gates, p_single, p_swap, invalid_qubits):
    random_circ = gen_random_list_circuit_alike(num_qubits, num_gates, p_single, p_swap)
    gate_num = len(random_circ)
    return (gate_num - len(remove_dead_gates_from_pseudo_list(random_circ, invalid_qubits)))

def time_consumed_in_dead_gate_removal_of_a_random_circuit(num_qubits, num_gates, p_single, p_swap, invalid_qubits):
    random_circ = gen_random_list_circuit_alike(num_qubits, num_gates, p_single, p_swap)
    start = time.time()
    remove_dead_gates_from_pseudo_list(random_circ, invalid_qubits)
    end = time.time()
    return 1000*(end - start)
