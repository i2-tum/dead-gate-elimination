xpoints_arr_lowsingle_singlequbit_time = []
ypoints_arr_lowsingle_singlequbit_time = []
xpoints_arr_lowsingle_doublequbit_time = []
ypoints_arr_lowsingle_doublequbit_time = []
xpoints_arr_lowsingle_triplequbit_time = []
ypoints_arr_lowsingle_triplequbit_time = []
ypoints_arr_lowsingle_10perqubit_time = []
ypoints_arr_lowsingle_20perqubit_time = []


for size in range(2, 100):
    
    print("Progress report:", end = ' ')
    print("Calculating the", size, end='')
    print("-th case.", end="\r")
    
    xpoints_arr_lowsingle_singlequbit_time.append(size)
    average_vect_lowsingle_singlequbit_time = []
    average_vect_lowsingle_doublequbit_time = []
    average_vect_lowsingle_triplequbit_time = []
    average_vect_lowsingle_10perqubit_time = []
    average_vect_lowsingle_20perqubit_time = []
    average_size = 10
    for t in range(average_size):
        average_vect_lowsingle_singlequbit_time.append(time_consumed_in_dead_gate_removal_of_a_random_circuit(size, 100*size, 0.1, 0.1, [0]))
        average_vect_lowsingle_doublequbit_time.append(time_consumed_in_dead_gate_removal_of_a_random_circuit(size, 100*size, 0.1, 0.1, [0, 1]))
        average_vect_lowsingle_triplequbit_time.append(time_consumed_in_dead_gate_removal_of_a_random_circuit(size, 100*size, 0.1, 0.1, [0, 1, 2]))
        l_10per = []
        l_20per = []
        for e in range(int(0.1*size)):
            l_10per.append(e)

        average_vect_lowsingle_10perqubit_time.append(time_consumed_in_dead_gate_removal_of_a_random_circuit(size, 100*size, 0.1, 0.1, l_10per))
        for e in range(int(0.2*size)):
            l_20per.append(e)
        average_vect_lowsingle_20perqubit_time.append(time_consumed_in_dead_gate_removal_of_a_random_circuit(size, 100*size, 0.1, 0.1, l_20per))
        


    ypoints_arr_lowsingle_singlequbit_time.append(np.mean(average_vect_lowsingle_singlequbit_time))
    ypoints_arr_lowsingle_doublequbit_time.append(np.mean(average_vect_lowsingle_doublequbit_time))
    ypoints_arr_lowsingle_triplequbit_time.append(np.mean(average_vect_lowsingle_triplequbit_time))
    ypoints_arr_lowsingle_10perqubit_time.append(np.mean(average_vect_lowsingle_10perqubit_time))
    ypoints_arr_lowsingle_20perqubit_time.append(np.mean(average_vect_lowsingle_20perqubit_time))
    

xpoints_lowsingle_singlequbit_time = np.array(xpoints_arr_lowsingle_singlequbit_time)

ypoints_lowsingle_singlequbit_time = np.array(ypoints_arr_lowsingle_singlequbit_time)
ypoints_lowsingle_doublequbit_time = np.array(ypoints_arr_lowsingle_doublequbit_time)
ypoints_lowsingle_triplequbit_time = np.array(ypoints_arr_lowsingle_triplequbit_time)
ypoints_lowsingle_10perqubit_time = np.array(ypoints_arr_lowsingle_10perqubit_time)
ypoints_lowsingle_20perqubit_time = np.array(ypoints_arr_lowsingle_20perqubit_time)


plt.grid()


plt.plot(xpoints_lowsingle_singlequbit_time, ypoints_lowsingle_singlequbit_time, label="1 dead qubit")
plt.plot(xpoints_lowsingle_singlequbit_time, ypoints_lowsingle_doublequbit_time, label="2 dead qubits")
plt.plot(xpoints_lowsingle_singlequbit_time, ypoints_lowsingle_triplequbit_time, label="3 dead qubits")
plt.plot(xpoints_lowsingle_singlequbit_time, ypoints_lowsingle_10perqubit_time, label="10% dead qubits", linestyle="-.")
plt.plot(xpoints_lowsingle_singlequbit_time, ypoints_lowsingle_20perqubit_time, label="20% dead qubits", linestyle=":")

plt.legend()

plt.show()
