import random
import numpy as np

def main():
    n = 10
    vaccination_percentage = 0.5
    transmission_chance = 0.1
    recovery_chance = 0.1
    connection_forming_chance = 1

    previous_status = np.zeros(n, dtype=int)

    intercourse_chart = np.zeros((n, n), dtype=int)

    # Fill the intercourse chart
    fill_intercourse_array(intercourse_chart, n, connection_forming_chance)
    print_array(intercourse_chart, n)

    # Fill the status array
    fill_status_array(previous_status, n, vaccination_percentage)
    print_array(previous_status, n)

    # Run the simulation
    infection_counts = run_simulation(transmission_chance, recovery_chance, intercourse_chart, previous_status, n)
    print(infection_counts)

def run_simulation(transmission_chance, recovery_chance, intercourse_chart, previous_status, n):
    infection_counts = []
    max_day = 1000
    day = 1
    current_status = np.zeros(n, dtype=int)

    while day <= max_day:
        all_recovered = True
        current_status = np.copy(previous_status)
        for i in range(n):
            for j in range(n):
                if (intercourse_chart[i][j] == 1 and previous_status[i] != previous_status[j] and previous_status[i] != -1 and previous_status[j] != -1):
                    r = random.random()
                    if r < transmission_chance:
                        current_status[i] = 1
                        current_status[j] = 1

        infected_count = 0
        for i in range(n):
            if current_status[i] == 1:
                r = random.random()
                if r < recovery_chance:
                    current_status[i] = 0

            if current_status[i] == 1:
                all_recovered = False
                infected_count += 1

        infection_counts.append(infected_count)
        previous_status = np.copy(current_status)
        if all_recovered:
            break
        day += 1

    # Add zeros for remaining days if all_recovered
    while day <= max_day:
        infection_counts.append(0)
        day += 1

    return infection_counts

def fill_intercourse_array(array, n, connection_forming_chance):
    for i in range(n):
        for j in range(n):
            if j < i:
                array[i][j] = 1 if random.random() < connection_forming_chance else 0
            else:
                array[i][j] = 0

def fill_status_array(array, n, vaccination_percentage):
    vaccination_count = round(vaccination_percentage * n)
    array.fill(0)  # Ensure array is initialized with zeros
    count = 0
    while count < vaccination_count:
        index = random.randint(0, n - 1)
        if array[index] != -1:
            array[index] = -1
            count += 1

    while True:
        index = random.randint(0, n - 1)
        if array[index] != -1:
            array[index] = 1
            break

def print_array(array, n):
    if isinstance(array, np.ndarray) and array.ndim == 2:
        print("Intercourse Chart:")
        for i in range(n):
            print(" ".join(map(str, array[i])))
    else:
        print("Status Array:")
        print(" ".join(map(str, array)))

if __name__ == "__main__":
    main()
