import random
import numpy as np
import pandas as pd
from tqdm import tqdm

#Time complexity: O(trials * n^2)
def main():
    trials = 1000 #Parameter: how many times the simulation should run 

    # Run the simulation using the top 5% super spreaders method
    result = run_super_spreaders(trials, "super_spreaders.csv")
    print(f"Super Spreaders Method:")
    print(f"Percentage of trials ending in zero infections: {result[0]}%")
    print(f"Average number of infected people during the stable phase: {result[2]}")

    # Run the simulation using the random vaccination method
    result = run_random_vaccination(trials, 0.5, "random_vaccination.csv") #Parameter: percentage of population to vaccinate
    print(f"Random Vaccination Method:")
    print(f"Percentage of trials ending in zero infections: {result[0]}%")
    print(f"Average number of infected people during the stable phase: {result[2]}")

#method of running the simulation using the top 5% super spreaders method
def run_super_spreaders(trials, filename):
    n = 100
    transmission_chance, recovery_chance, connection_forming_chance = 0.1, 0.1, 0.5 #Parameters: transmission chance, recovery chance, connection forming chance

    previous_status = np.zeros(n, dtype=int)
    connect_amount = np.zeros(n, dtype=int)
    super_spreaders = np.zeros(n, dtype=int)
    intercourse_chart = np.zeros((n, n), dtype=int)

    fill_intercourse_array(intercourse_chart, n, connection_forming_chance)
    fill_connect_amount(connect_amount, n, intercourse_chart)
    fill_super_spreaders(super_spreaders, n, connect_amount)
    fill_status_array_with_super_spreaders(previous_status, n, super_spreaders)

    return run_trials(trials, transmission_chance, recovery_chance, intercourse_chart, previous_status, n, filename)

#method of running the simulation using the random vaccination method
def run_random_vaccination(trials, vaccination_percentage, filename):
    n = 100
    transmission_chance, recovery_chance, connection_forming_chance = 0.1, 0.1, 0.5 #Parameters: transmission chance, recovery chance, connection forming chance

    previous_status = np.zeros(n, dtype=int)
    connect_amount = np.zeros(n, dtype=int)
    intercourse_chart = np.zeros((n, n), dtype=int)

    fill_intercourse_array(intercourse_chart, n, connection_forming_chance)
    fill_connect_amount(connect_amount, n, intercourse_chart)
    random_fill_status_array(previous_status, n, vaccination_percentage)

    return run_trials(trials, transmission_chance, recovery_chance, intercourse_chart, previous_status, n, filename)

#method of running the simulation muktiple times and obtain key data
def run_trials(trials, transmission_chance, recovery_chance, intercourse_chart, previous_status, n, filename):
    end_in_zero_count = 0
    #array used to calculate the stable position of the infection given parameters
    stable_infections = []
    results = []

    #progress bar 
    for trial in tqdm(range(trials), desc="Running Trials"):
        infection_counts = run_simulation(transmission_chance, recovery_chance, intercourse_chart, previous_status, n)
        trial_result = {
            "trial": trial + 1,
            "end_in_zero": infection_counts[-1] == 0,
            "daily_infections": infection_counts
        }
        if infection_counts[-1] == 0:
            end_in_zero_count += 1
        else:
            #apply kalman filter to filter out the noise
            stable_infections.append(apply_kalman_filter(infection_counts))
        results.append(trial_result)

    stable_avg = np.mean(stable_infections) if stable_infections else 0
    end_in_zero_percentage = end_in_zero_count / trials * 100

    # Save the results to a CSV file
    save_results_to_csv(results, filename)

    return end_in_zero_percentage, results[0]['daily_infections'], stable_avg

#method for each iteration
def run_simulation(transmission_chance, recovery_chance, intercourse_chart, previous_status, n):
    infection_counts = []
    max_day = 1000 #Parameter: maximum number of days in a single simulation
    current_status = np.zeros(n, dtype=int)

    for day in range(1, max_day + 1):
        all_recovered = True
        current_status[:] = previous_status
        for i in range(n):
            for j in range(i):
                if intercourse_chart[i, j] == 1 and previous_status[i] != previous_status[j] and previous_status[i] != -1 and previous_status[j] != -1:
                    if random.random() < transmission_chance:
                        current_status[i] = current_status[j] = 1

        infected_count = 0
        for i in range(n):
            if current_status[i] == 1:
                if random.random() < recovery_chance:
                    current_status[i] = 0
            if current_status[i] == 1:
                all_recovered = False
                infected_count += 1

        infection_counts.append(infected_count)
        previous_status[:] = current_status
        if all_recovered:
            infection_counts.extend([0] * (max_day - day))
            break
    return infection_counts

#kalman filter  
def apply_kalman_filter(data):
    n_iter = len(data)
    Q, R = 1e-5, 0.01
    xhat, P, xhatminus, Pminus, K = np.zeros(n_iter), np.zeros(n_iter), np.zeros(n_iter), np.zeros(n_iter), np.zeros(n_iter)
    xhat[0], P[0] = data[0], 1.0

    for k in range(1, n_iter):
        xhatminus[k], Pminus[k] = xhat[k-1], P[k-1] + Q
        K[k] = Pminus[k] / (Pminus[k] + R)
        xhat[k] = xhatminus[k] + K[k] * (data[k] - xhatminus[k])
        P[k] = (1 - K[k]) * Pminus[k]

    return np.mean(xhat[-10:])

#ingtercourse chart rng 
def fill_intercourse_array(array, n, connection_forming_chance):
    for i in range(n):
        for j in range(i):
            array[i, j] = 1 if random.random() < connection_forming_chance else 0

#status array rng 
def random_fill_status_array(array, n, vaccination_percentage):
    vaccination_count = round(vaccination_percentage * n)
    array.fill(0)
    vaccinated_indices = np.random.choice(n, vaccination_count, replace=False)
    array[vaccinated_indices] = -1
    non_vaccinated_indices = np.setdiff1d(np.arange(n), vaccinated_indices)
    array[np.random.choice(non_vaccinated_indices, 1)] = 1
    
#status array rng for super spreaders
def fill_status_array_with_super_spreaders(array, n, super_spreaders):
    array.fill(0)
    array[super_spreaders == 1] = -1
    non_super_spreaders_indices = np.where(super_spreaders == 0)[0]
    array[np.random.choice(non_super_spreaders_indices, 1)] = 1

#find amount of connections for each node 
def fill_connect_amount(array, n, intercourse_chart):
    array[:] = np.sum(intercourse_chart, axis=1) + np.sum(intercourse_chart, axis=0)

#find super spreaders based on connection 
def fill_super_spreaders(array, n, connect_amount):
    super_spreaders_count = round(0.05 * n)
    top_indices = np.argpartition(connect_amount, -super_spreaders_count)[-super_spreaders_count:]
    array.fill(0)
    array[top_indices] = 1

#export the results to a csv located in the same root folder as the script is in 
def save_results_to_csv(results, filename):
    data = []
    for result in results:
        for day, infections in enumerate(result['daily_infections']):
            data.append({
                "trial": result["trial"],
                "day": day + 1,
                "infections": infections
            })
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)

#If run, run main method
if __name__ == "__main__":
    main()
