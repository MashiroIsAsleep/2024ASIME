import random
import numpy as np
from tqdm import tqdm


def main():
    
    #Run the simulation using the top 5% super spreaders method
    # trials = 100  
    # print(run_super_spreaders(trials))
    
    #Run the simulation using the random vaccination method
    trials = 100
    vaccination_percentage = 0.5
    print(run_random_vaccination(trials, vaccination_percentage))
    
    #below are initializations might be useful for later
    # n = 100
    # vaccination_percentage = 0.5
    # transmission_chance = 0.1
    # recovery_chance = 0.1
    # connection_forming_chance = .5
    
    # previous_status = np.zeros(n, dtype=int)
    # connect_amount = np.zeros(n, dtype=int)
    # super_spreaders = np.zeros(n, dtype=int)
    
    # intercourse_chart = np.zeros((n, n), dtype=int)

    # Fill the intercourse chart
    #fill_intercourse_array(intercourse_chart, n, connection_forming_chance)
    #print_array(intercourse_chart, n)

    # Fill the status array randomly
    #random_fill_status_array(previous_status, n, vaccination_percentage)
    #print_array(previous_status, n)
    
    #vaccinate the top 5% superspreaders
    #fill_status_array_with_super_spreaders(previous_status, n, super_spreaders) 
    #print_array(previous_status, n)
    
    # Fill the connect amount array
    #fill_connect_amount(connect_amount, n, intercourse_chart)
    #print_array(connect_amount, n)

    # Fill the super spreaders array
    #fill_super_spreaders(super_spreaders, n, connect_amount)
    #print_array(super_spreaders, n)
    
    # Run the simulation
    #infection_counts = run_simulation(transmission_chance, recovery_chance,
    #                                  intercourse_chart, previous_status, n)
    #print(infection_counts)


def run_super_spreaders(trials):
    n = 100
    transmission_chance = 0.1
    recovery_chance = 0.1
    connection_forming_chance = .5
    
    previous_status = np.zeros(n, dtype=int)
    connect_amount = np.zeros(n, dtype=int)
    super_spreaders = np.zeros(n, dtype=int)
    
    intercourse_chart = np.zeros((n, n), dtype=int)

    fill_intercourse_array(intercourse_chart, n, connection_forming_chance)
    fill_connect_amount(connect_amount, n, intercourse_chart)
    fill_super_spreaders(super_spreaders, n, connect_amount)
    fill_status_array_with_super_spreaders(previous_status, n, super_spreaders)  
    
    end_in_zero_count = 0
    for _ in tqdm(range(trials), desc="Running Trials"):
        infection_counts = run_simulation(transmission_chance, recovery_chance,
                                          intercourse_chart, previous_status, n) 
        if infection_counts[-1] == 0:
            end_in_zero_count += 1
    return end_in_zero_count / trials * 100
    
def run_random_vaccination(trials, vaccination_percentage):
    n = 100
    transmission_chance = 0.1
    recovery_chance = 0.1
    connection_forming_chance = .5
    
    previous_status = np.zeros(n, dtype=int)
    connect_amount = np.zeros(n, dtype=int)
    
    intercourse_chart = np.zeros((n, n), dtype=int)

    fill_intercourse_array(intercourse_chart, n, connection_forming_chance)
    fill_connect_amount(connect_amount, n, intercourse_chart)
    random_fill_status_array(previous_status, n, vaccination_percentage)
    
    end_in_zero_count = 0
    for _ in tqdm(range(trials), desc="Running Trials"):
        infection_counts = run_simulation(transmission_chance, recovery_chance,
                                          intercourse_chart, previous_status, n) 
        if infection_counts[-1] == 0:
            end_in_zero_count += 1
    return end_in_zero_count / trials * 100



def run_simulation(transmission_chance, recovery_chance, intercourse_chart,
                   previous_status, n):
    infection_counts = []
    max_day = 1000
    day = 1
    current_status = np.zeros(n, dtype=int)

    while day <= max_day:
        all_recovered = True
        current_status = np.copy(previous_status)
        for i in range(n):
            for j in range(n):
                if (intercourse_chart[i][j] == 1
                        and previous_status[i] != previous_status[j]
                        and previous_status[i] != -1
                        and previous_status[j] != -1):
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
                array[i][j] = 1 if random.random(
                ) < connection_forming_chance else 0
            else:
                array[i][j] = 0


def random_fill_status_array(array, n, vaccination_percentage):
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

def fill_status_array_with_super_spreaders(array, n, super_spreaders):
    for i in range(n):
        if super_spreaders[i] == 1:
            array[i] = -1
    while True:
        index = random.randint(0, n - 1)
        if array[index] != -1:
            array[index] = 1
            break 

def fill_connect_amount(array, n, intercourse_chart):
    for i in range(n):
        sum = 0
        for j in range(n):
            sum += intercourse_chart[i][j]
            sum += intercourse_chart[j][i]
        array[i] = sum

def fill_super_spreaders(array, n, connect_amount):
    super_spreaders_percentage = 0.05
    super_spreaders_count = round(super_spreaders_percentage * n)
    indices = np.argpartition(connect_amount, -super_spreaders_count)[-super_spreaders_count:]
    indices_sorted = indices[np.argsort(-connect_amount[indices])]
    array.fill(0)
    for i in range(super_spreaders_count):
        array[indices_sorted[i]] = 1
    
    
def print_array(array, n):
    if isinstance(array, np.ndarray) and array.ndim == 2:
        print("2D Chart:")
        for i in range(n):
            print(" ".join(map(str, array[i])))
    else:
        print("Array:")
        print(" ".join(map(str, array)))


if __name__ == "__main__":
    main()
