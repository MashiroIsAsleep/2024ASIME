import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

def main():
    trials = 300  # Number of trials for better parallelism
    n = 500
    transmission_chance, recovery_chance, connection_forming_chance = 0.5, 0.15, 0.004

    # Create results directory if it does not exist
    os.makedirs("results", exist_ok=True)

    # Run the simulation using the top 5% super spreaders method
    result = run_trials_super_spreaders(trials, transmission_chance, recovery_chance, connection_forming_chance, n, "results/av2partner/super_spreaders.csv")
    print(f"Super Spreaders Method:")
    print(f"Percentage of trials ending in zero infections: {result[0]}%")

    # Run the simulation using the random vaccination method
    for i in range(30):
        p = 0.15 + i * 0.005
        filename = f"results/av2partner/random_vaccination_{p*100:.2f}.csv"
        result = run_trials_random_vaccination(trials, transmission_chance, recovery_chance, connection_forming_chance, p, n, filename)
        print(f"Random Vaccination Method for {p*100:.2f}% of the population:")
        print(f"Percentage of trials ending in zero infections: {result[0]}%")

def run_trials_super_spreaders(trials, transmission_chance, recovery_chance, connection_forming_chance, n, filename):
    # Using joblib to parallelize the trials
    results = Parallel(n_jobs=-3)(delayed(run_single_trial_super_spreaders)(trial_num, transmission_chance, recovery_chance, connection_forming_chance, n) for trial_num in tqdm(range(1, trials + 1), desc="Running Trials"))

    end_in_zero_count = sum(1 for result in results if result['end_in_zero'])
    end_in_zero_percentage = end_in_zero_count / trials * 100

    # Save the results to a CSV file
    save_results_to_csv(results, filename)

    return end_in_zero_percentage, results[0]['daily_infections']

def run_trials_random_vaccination(trials, transmission_chance, recovery_chance, connection_forming_chance, vaccination_percentage, n, filename):
    # Using joblib to parallelize the trials
    results = Parallel(n_jobs=-3)(delayed(run_single_trial_random_vaccination)(trial_num, transmission_chance, recovery_chance, connection_forming_chance, vaccination_percentage, n) for trial_num in tqdm(range(1, trials + 1), desc="Running Trials"))

    end_in_zero_count = sum(1 for result in results if result['end_in_zero'])
    end_in_zero_percentage = end_in_zero_count / trials * 100

    # Save the results to a CSV file
    save_results_to_csv(results, filename)

    return end_in_zero_percentage, results[0]['daily_infections']

def run_single_trial_super_spreaders(trial_num, transmission_chance, recovery_chance, connection_forming_chance, n):
    previous_status_ss = np.zeros(n, dtype=int)
    connect_amount = np.zeros(n, dtype=int)
    super_spreaders = np.zeros(n, dtype=int)
    intercourse_chart = np.zeros((n, n), dtype=int)

    fill_intercourse_array(intercourse_chart, n, connection_forming_chance)
    connect_amount[:] = np.sum(intercourse_chart, axis=1) + np.sum(intercourse_chart, axis=0)
    fill_super_spreaders(super_spreaders, n, connect_amount)
    fill_status_array_with_super_spreaders(previous_status_ss, n, super_spreaders)

    infection_counts = run_simulation(transmission_chance, recovery_chance, intercourse_chart, previous_status_ss, n)
    trial_result = {
        "trial": trial_num,
        "end_in_zero": infection_counts[-1] == 0,
        "daily_infections": infection_counts
    }
    return trial_result

def run_single_trial_random_vaccination(trial_num, transmission_chance, recovery_chance, connection_forming_chance, vaccination_percentage, n):
    previous_status_rv = np.zeros(n, dtype=int)
    intercourse_chart = np.zeros((n, n), dtype=int)

    fill_intercourse_array(intercourse_chart, n, connection_forming_chance)
    random_fill_status_array(previous_status_rv, n, vaccination_percentage)

    infection_counts = run_simulation(transmission_chance, recovery_chance, intercourse_chart, previous_status_rv, n)
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
        
        # Transmission based on intercourse chart
        transmission_matrix = (np.random.rand(n, n) < transmission_chance) & (intercourse_chart == 1)
        new_infections = (transmission_matrix & (previous_status == 1)[:, None]) | (transmission_matrix & (previous_status == 1)[None, :])
        
        # Ensure vaccinated individuals (-1) do not get infected
        new_infections &= (previous_status != -1)[:, None]
        current_status = np.maximum(current_status, new_infections.any(axis=0).astype(int))

        # Recovery process
        recovery_matrix = (current_status == 1) & (np.random.rand(n) < recovery_chance)
        current_status[recovery_matrix] = 0

        # Ensure vaccinated individuals (-1) remain unchanged
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

def random_fill_status_array(array, n, vaccination_percentage):
    vaccination_count = round(vaccination_percentage * n)
    array.fill(0)
    vaccinated_indices = np.random.choice(n, vaccination_count, replace=False)
    array[vaccinated_indices] = -1
    non_vaccinated_indices = np.setdiff1d(np.arange(n), vaccinated_indices)
    array[np.random.choice(non_vaccinated_indices, 1)] = 1

def fill_status_array_with_super_spreaders(array, n, super_spreaders):
    array.fill(0)
    array[super_spreaders == 1] = -1
    non_super_spreaders_indices = np.where(super_spreaders == 0)[0]
    array[np.random.choice(non_super_spreaders_indices, 1)] = 1

def fill_super_spreaders(array, n, connect_amount):
    super_spreaders_count = round(0.05 * n)
    top_indices = np.argpartition(connect_amount, -super_spreaders_count)[-super_spreaders_count:]
    array.fill(0)
    array[top_indices] = 1

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
