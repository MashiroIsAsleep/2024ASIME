import random
import numpy as np
from tqdm import tqdm

def main():
    # Run the simulation using the top 5% super spreaders method
    trials = 1
    result = run_super_spreaders(trials)
    print(f"Percentage of trials ending in zero infections: {result[0]}%")
    print("Number of people infected each day (example from first trial):")
    print(result[1])
    print(f"Average number of infected people during the stable phase: {result[2]}")

    # Run the simulation using the random vaccination method
    trials = 1
    vaccination_percentage = 0.5
    result = run_random_vaccination(trials, vaccination_percentage)
    print(f"Percentage of trials ending in zero infections: {result[0]}%")
    print("Number of people infected each day (example from first trial):")
    print(result[1])
    print(f"Average number of infected people during the stable phase: {result[2]}")

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
    stable_infections = []
    daily_infections = None
    for _ in tqdm(range(trials), desc="Running Trials"):
        infection_counts = run_simulation(transmission_chance, recovery_chance,
                                          intercourse_chart, previous_status, n) 
        if daily_infections is None:
            daily_infections = infection_counts
        if infection_counts[-1] == 0:
            end_in_zero_count += 1
        else:
            stable_infections.append(apply_kalman_filter(infection_counts))
    stable_avg = np.mean(stable_infections) if stable_infections else 0
    return end_in_zero_count / trials * 100, daily_infections, stable_avg

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
    stable_infections = []
    daily_infections = None
    for _ in tqdm(range(trials), desc="Running Trials"):
        infection_counts = run_simulation(transmission_chance, recovery_chance,
                                          intercourse_chart, previous_status, n) 
        if daily_infections is None:
            daily_infections = infection_counts
        if infection_counts[-1] == 0:
            end_in_zero_count += 1
        else:
            stable_infections.append(apply_kalman_filter(infection_counts))
    stable_avg = np.mean(stable_infections) if stable_infections else 0
    return end_in_zero_count / trials * 100, daily_infections, stable_avg

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

def apply_kalman_filter(data):
    # Kalman filter parameters
    n_iter = len(data)
    sz = (n_iter,) # size of array
    Q = 1e-5 # process variance

    # allocate space for arrays
    xhat = np.zeros(sz)      # a posteri estimate of x
    P = np.zeros(sz)         # a posteri error estimate
    xhatminus = np.zeros(sz) # a priori estimate of x
    Pminus = np.zeros(sz)    # a priori error estimate
    K = np.zeros(sz)         # gain or blending factor

    R = 0.1**2 # estimate of measurement variance, change to see effect

    # initial guesses
    xhat[0] = data[0]
    P[0] = 1.0

    for k in range(1,n_iter):
        # time update
        xhatminus[k] = xhat[k-1]
        Pminus[k] = P[k-1]+Q

        # measurement update
        K[k] = Pminus[k]/(Pminus[k]+R)
        xhat[k] = xhatminus[k]+K[k]*(data[k]-xhatminus[k])
        P[k] = (1-K[k])*Pminus[k]

    # Return the average of the stable portion of the filtered data
    stable_data = xhat[-10:]  # assuming last 10 points represent stable state
    return np.mean(stable_data)

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
