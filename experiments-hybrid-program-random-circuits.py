xpoints_arr_lowsingle_singlequbit = []
ypoints_arr_lowsingle_singlequbit = []
xpoints_arr_lowsingle_doublequbit = []
ypoints_arr_lowsingle_doublequbit = []
xpoints_arr_lowsingle_triplequbit = []
ypoints_arr_lowsingle_triplequbit = []
ypoints_arr_lowsingle_10perqubit = []
ypoints_arr_lowsingle_20perqubit = []

n_blocks = 60
p_single_gate = 0.1
p_swap = 0.1

for size in range(2, 100):
    
    print("Progress report:", end = ' ')
    print("Calculating the", size, end='')
    print("-th case.", end="\r")
    
    l_10per = []
    l_20per = []
    for e in range(int(0.1*size)):
        l_10per.append(e)
    for e in range(int(0.2*size)):
        l_20per.append(e)
        
    
    xpoints_arr_lowsingle_singlequbit.append(size)
    average_vect_lowsingle_singlequbit = []
    average_vect_lowsingle_doublequbit = []
    average_vect_lowsingle_triplequbit = []
    average_vect_lowsingle_10perqubit = []
    average_vect_lowsingle_20perqubit = []
    average_size = 100
    for t in range(average_size):
        
        ### Init average_vect ###
        average_vect_lowsingle_singlequbit.append(0)
        average_vect_lowsingle_doublequbit.append(0)
        average_vect_lowsingle_triplequbit.append(0)
        average_vect_lowsingle_10perqubit.append(0)
        average_vect_lowsingle_20perqubit.append(0)

        for _ in range(n_blocks):
            average_vect_lowsingle_singlequbit[t] += num_removed_dead_gates_of_a_random_circuit(size, 100*size, p_single_gate, p_swap, [0])
            average_vect_lowsingle_doublequbit[t] += num_removed_dead_gates_of_a_random_circuit(size, 100*size, p_single_gate, p_swap, [0, 1])
            average_vect_lowsingle_triplequbit[t] += num_removed_dead_gates_of_a_random_circuit(size, 100*size, p_single_gate, p_swap, [0, 1, 2])
            average_vect_lowsingle_10perqubit[t] += num_removed_dead_gates_of_a_random_circuit(size, 100*size, p_single_gate, p_swap, l_10per.copy())
            average_vect_lowsingle_20perqubit[t] += num_removed_dead_gates_of_a_random_circuit(size, 100*size, p_single_gate, p_swap, l_20per.copy())


    ypoints_arr_lowsingle_singlequbit.append(np.mean(average_vect_lowsingle_singlequbit))
    ypoints_arr_lowsingle_doublequbit.append(np.mean(average_vect_lowsingle_doublequbit))
    ypoints_arr_lowsingle_triplequbit.append(np.mean(average_vect_lowsingle_triplequbit))
    ypoints_arr_lowsingle_10perqubit.append(np.mean(average_vect_lowsingle_10perqubit))
    ypoints_arr_lowsingle_20perqubit.append(np.mean(average_vect_lowsingle_20perqubit))
    
# ypoints_highsingle_singlequbit = np.array(ypoints_arr_highsingle_singlequbit)

xpoints_lowsingle_singlequbit = np.array(xpoints_arr_lowsingle_singlequbit)

ypoints_lowsingle_singlequbit = np.array(ypoints_arr_lowsingle_singlequbit)
ypoints_lowsingle_doublequbit = np.array(ypoints_arr_lowsingle_doublequbit)
ypoints_lowsingle_triplequbit = np.array(ypoints_arr_lowsingle_triplequbit)

ypoints_lowsingle_10perqubit = np.array(ypoints_arr_lowsingle_10perqubit)
ypoints_lowsingle_20perqubit = np.array(ypoints_arr_lowsingle_20perqubit)



plt.grid()


plt.plot(xpoints_lowsingle_singlequbit, ypoints_lowsingle_singlequbit, label="1 dead qubit")
plt.plot(xpoints_lowsingle_singlequbit, ypoints_lowsingle_doublequbit, label="2 dead qubits")
plt.plot(xpoints_lowsingle_singlequbit, ypoints_lowsingle_triplequbit, label="3 dead qubits")
plt.plot(xpoints_lowsingle_singlequbit, ypoints_lowsingle_10perqubit, label="10% dead qubits", linestyle="-.")
plt.plot(xpoints_lowsingle_singlequbit, ypoints_lowsingle_20perqubit, label="20% dead qubits", linestyle=":")

plt.legend()

plt.show()
