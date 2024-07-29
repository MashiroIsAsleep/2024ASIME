import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

def main():
    trials = 300  # Number of trials for better parallelism
    n = 500
    transmission_chance, recovery_chance, connection_forming_chance = 0.5, 0.15, 0.012

    # Create results directory if it does not exist
    os.makedirs("results", exist_ok=True)

    # Run the simulation using the top 5% super spreaders method
    result = run_trials(trials, transmission_chance, recovery_chance, connection_forming_chance, n, "results/super_spreaders.csv", method="super_spreaders")
    print(f"Super Spreaders Method:")
    print(f"Percentage of trials ending in zero infections: {result[0]}%")

    # Uncomment to run the simulation using the random vaccination method
    # for i in range(30):
    #     p = + i * 0.005
    #     filename = f"results/random_vaccination_{p*100:.2f}.csv"
    #     result = run_trials(trials, transmission_chance, recovery_chance, connection_forming_chance, n, filename, vaccination_percentage=p, method="random_vaccination")
    #     print(f"Random Vaccination Method for {p*100:.2f}% of the population:")
    #     print(f"Percentage of trials ending in zero infections: {result[0]}%")

def run_trials(trials, transmission_chance, recovery_chance, connection_forming_chance, n, filename, method="super_spreaders", vaccination_percentage=0):
    results = Parallel(n_jobs=-1)(delayed(run_single_trial)(trial_num, transmission_chance, recovery_chance, connection_forming_chance, n, method, vaccination_percentage) for trial_num in tqdm(range(1, trials + 1), desc="Running Trials"))

    end_in_zero_count = sum(1 for result in results if result['end_in_zero'])
    end_in_zero_percentage = end_in_zero_count / trials * 100

    save_results_to_csv(results, filename)

    return end_in_zero_percentage, results[0]['daily_infections']

def run_single_trial(trial_num, transmission_chance, recovery_chance, connection_forming_chance, n, method, vaccination_percentage):
    previous_status = np.zeros(n, dtype=int)
    intercourse_chart = np.zeros((n, n), dtype=int)

    fill_intercourse_array(intercourse_chart, n, connection_forming_chance)

    if method == "super_spreaders":
        connect_amount = np.sum(intercourse_chart, axis=1) + np.sum(intercourse_chart, axis=0)
        super_spreaders = np.zeros(n, dtype=int)
        fill_super_spreaders(super_spreaders, n, connect_amount)
    initialize_status_array(previous_status, n)

    infection_counts = run_simulation(transmission_chance, recovery_chance, intercourse_chart, previous_status, n)
    trial_result = {
        "trial": trial_num,
        "end_in_zero": infection_counts[-1] == 0,
        "daily_infections": infection_counts
    }
    return trial_result

def run_simulation(transmission_chance, recovery_chance, intercourse_chart, previous_status, n):
    infection_counts = []
    max_day = 200

    for day in range(1, max_day + 1):
        current_status = previous_status.copy()
        
        transmission_matrix = (np.random.rand(n, n) < transmission_chance) & (intercourse_chart == 1)
        new_infections = (transmission_matrix & (previous_status == 1)[:, None]) | (transmission_matrix & (previous_status == 1)[None, :])
        new_infections &= (previous_status != -1)[:, None]
        current_status = np.maximum(current_status, new_infections.any(axis=0).astype(int))

        recovery_matrix = (current_status == 1) & (np.random.rand(n) < recovery_chance)
        current_status[recovery_matrix] = 0
        current_status[previous_status == -1] = -1

        infected_count = np.sum(current_status == 1)
        infection_counts.append(infected_count)

        if infected_count == 0:
            infection_counts.extend([0] * (max_day - day))
            break

        previous_status[:] = current_status
    return infection_counts

def fill_intercourse_array(array, n, connection_forming_chance):
    random_matrix = np.random.rand(n, n)
    array[:] = (random_matrix < connection_forming_chance).astype(int)
    np.fill_diagonal(array, 0)

def fill_super_spreaders(array, n, connect_amount):
    super_spreaders_count = round(0.05 * n)
    top_indices = np.argpartition(connect_amount, -super_spreaders_count)[-super_spreaders_count:]
    array.fill(0)
    array[top_indices] = 1
    
def initialize_status_array(array, n):
    array[np.random.choice(np.arange(n), 1)] = 1

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

if __name__ == "__main__":
    main()
