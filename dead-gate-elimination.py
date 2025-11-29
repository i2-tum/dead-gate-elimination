from qiskit import QuantumCircuit
from typing import List, Dict

# Assuming 'find_dead_outcomes' is already implemented and accessible
# def find_dead_outcomes(py_src: str, func_name: str, measurement_vars: List[str]) -> List[str]:
#     """
#     This function should be implemented to perform the static analysis.
#     It returns a list of 'dead' qubits/measurement outcomes that are non-contributory.
#     """
#     pass

def frontier(circuit: QuantumCircuit) -> List:
    """
    This function tracks the frontier gates (those that affect the outcome of the circuit).
    
    Parameters:
        circuit (QuantumCircuit): The quantum circuit to analyze.
    
    Returns:
        List: A list of frontier gates.
    """
    frontier_gates = []
    for gate in circuit.data:
        if any(qbit in gate[1] for qbit in circuit.qubits):
            frontier_gates.append(gate)
    return frontier_gates

def convert_classical_to_qubits(classical_vars: List[str], measurement_map: Dict[str, str]) -> List[str]:
    """
    Convert a list of classical variables into their corresponding qubits using the measurement_map.
    
    Parameters:
        classical_vars (List[str]): List of classical variables (e.g., ['a', 'b', 'c']).
        measurement_map (Dict[str, str]): Mapping of qubits to classical variables (e.g., {'q[0]': 'a', 'q[1]': 'b'}).
    
    Returns:
        List[str]: List of qubits corresponding to the classical variables (e.g., ['q[0]', 'q[1]']).
    """
    # Reverse the mapping of measurement_map to look up qubits by classical variables
    reversed_map = {v: k for k, v in measurement_map.items()}
    
    # Convert classical variables to qubits using the reversed mapping
    qubits = [reversed_map[var] for var in classical_vars if var in reversed_map]
    
    return qubits

def frontier(circuit: QuantumCircuit) -> List:
    """
    This function tracks the frontier gates (those that affect the outcome of the circuit).
    
    Parameters:
        circuit (QuantumCircuit): The quantum circuit to analyze.
    
    Returns:
        List: A list of frontier gates.
    """
    frontier_gates = []
    for gate in circuit.data:
        if any(qbit in gate[1] for qbit in circuit.qubits):
            frontier_gates.append(gate)
    return frontier_gates

def is_dead_gate(gate, dead_outcomes: list, measurement_map: dict, circuit: QuantumCircuit) -> bool:
    """
    Check whether a given gate from a Qiskit QuantumCircuit is "dead",
    meaning it acts only on qubits whose measurement outcomes were determined
    to be non‑contributory (i.e., dead_outcomes).

    Parameters:
        gate: a CircuitInstruction (instruction record) from circuit.data
        dead_outcomes: list of classical variable names (strings) determined to be dead
        measurement_map: dict mapping qubit_label -> classical variable name
                         e.g. {'q[0]': 'a', 'q[1]': 'b'}
        circuit: the QuantumCircuit object in which the gate appears

    Returns:
        bool: True if this gate can be considered dead (and thus removable), False otherwise.
    """
    # Step 1: convert dead classical variables into corresponding qubit labels
    dead_qubit_labels = convert_classical_to_qubits(dead_outcomes, measurement_map)
    # Debug print
#     print(f"Dead classical outcomes: {dead_outcomes}")
#     print(f"Mapped to dead qubits (labels): {dead_qubit_labels}")

    # Step 2: check each qubit argument of the gate
    dead_qubits_involved = []
    for qubit in gate.qubits:  # gate.qubits is a tuple of Qubit objects (or Bit)
        # Use circuit.find_bit() to find register name and index for this qubit
        bit_loc = circuit.find_bit(qubit)  # returns namedtuple(index, registers)
        # registers is a list of (Register, index_in_that_register) pairs
        reg, idx = bit_loc.registers[0]  # pick first register info
        qubit_label = f"{reg.name}[{idx}]"
        # Debug print
#         print(f"Checking qubit object {qubit} → label '{qubit_label}'")
        if qubit_label in dead_qubit_labels:
            dead_qubits_involved.append(qubit_label)
    
    # Step 3: report result
    if dead_qubits_involved:
#         print(f"Gate {gate} involves dead qubits: {dead_qubits_involved}")
        return True
    else:
#         print(f"Gate {gate} does NOT involve dead qubits.")
        return False



def simplify_quantum_circuit(circuit_str: str, classical_func_str: str, measurement_map: Dict[str, str]) -> QuantumCircuit:
    """
    Optimize a quantum circuit by removing dead gates associated with non-contributory measurement outcomes.
    
    Parameters:
        circuit_str (str): Quantum circuit as a string in Qiskit format (QASM).
        classical_func_str (str): Classical function as a string for static analysis.
        measurement_map (Dict[str, str]): A mapping from measurement outcomes to classical variables.
    
    Returns:
        QuantumCircuit: The optimized quantum circuit object.
    """
    # Step 1: Identify the dead outcomes from the classical function using find_dead_outcomes
    measurement_vars = list(measurement_map.values())  # Get the classical variables from the map
    
    dead_outcomes = find_dead_outcomes(classical_func_str, func_name="prog", measurement_vars=measurement_vars)
    print(f"Dead outcomes: {dead_outcomes}")  # Debugging: Check if dead outcome is identified correctly
    
    if dead_outcomes is None:
        raise ValueError("find_dead_outcomes returned None, which is not valid.")
    
    # Step 2: Parse the input circuit using Qiskit (correct method)
    try:
        circuit = QuantumCircuit.from_qasm_str(circuit_str)  # Correct method for parsing QASM string
    except Exception as e:
        raise ValueError(f"Error parsing the QASM string: {str(e)}")
    
    # Step 3: Optimize the circuit by removing dead gates based on the dead outcomes
    optimized_circuit = circuit.copy()  # Copy the original circuit to avoid modification
    terminate = False
    
    while not terminate:
        terminate = True
        # Step 3.1: Check the frontier of gates that haven't been processed yet
        frontier_gates = frontier(optimized_circuit)
        
        for gate in frontier_gates:
            # Step 3.2: Check if the gate is a dead gate (using Theorems IV.1, IV.2, IV.3)
            if is_dead_gate(gate, dead_outcomes, measurement_map, optimized_circuit):  # Pass the circuit here
#                 print(f"Removing gate: {gate}")  # Debugging statement to check which gate is being removed
                # Remove the gate from the circuit
                optimized_circuit.data.remove(gate)
                terminate = False  # Indicate that further gates might be removable
                
    return optimized_circuit

# Example usage:

# Quantum circuit in QASM format
quantum_circuit_str = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[3];
    creg c[3];
    h q[0];
    cx q[1], q[0];
    ccx q[2], q[1], q[0];
    cx q[2], q[0];
    measure q[0] -> c[0];
    measure q[1] -> c[1];
    measure q[2] -> c[2];
"""

# Classical function that uses the measurement outcomes
classical_function_str_a = """
def prog(a, b, c):
    x = 2*a - 3*b
    y = b + c + 2
    z = x * y
    t = a * y
    return z - 2*t
"""
classical_function_str_b = """
def prog(a, b, c):
    i = random(0.1, 0.5)
    x = i * a
    y = b * c
    z = int(x + y)
    return z
"""

classical_function_str_c = """
def prog(a, b, c):
    if a:
        x = a + b
        y = c - a
        u = x * y
        v = a * (b - c)
        w = a * a
    else:
        i = random(0.1, 0.5)
        u = int(i*a + b*c)
        v = b - c
        w = c - b
    return u + v + w
"""

# Map measurement outcomes to classical variables
measurement_map = {
    'q[0]': 'a',  # outcome of q[0] is stored in variable 'a'
    'q[1]': 'b',  # outcome of q[1] is stored in variable 'b'
    'q[2]': 'c'   # outcome of q[2] is stored in variable 'c'
}

# Call the simplify_quantum_circuit function
optimized_circuit = simplify_quantum_circuit(quantum_circuit_str, classical_function_str_c, measurement_map)
print(optimized_circuit)
