import random
import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

def main():
    trials = 500  # Parameter: how many times the simulation should run

    # Run the simulation using the top 5% super spreaders method
    result = run_super_spreaders(trials, "super_spreaders.csv")
    print(f"Super Spreaders Method:")
    print(f"Percentage of trials ending in zero infections: {result[0]}%")
    print(f"Average number of infected people during the stable phase: {result[2]}")

    # Run the simulation using the random vaccination method
    vaccination_percentages = [i * 0.005 for i in range(21)]  # 0% to 10%
    results = []
    for p in vaccination_percentages:
        filename = f"random_vaccination_{p*100:.2f}.csv"
        result = run_random_vaccination(trials, p, filename)
        results.append((p, result[0], result[2]))
        print(f"Random Vaccination Method for {p*100:.2f}% of the population:")
        print(f"Percentage of trials ending in zero infections: {result[0]}%")
        print(f"Average number of infected people during the stable phase: {result[2]}")

    # Plot the results
    plot_results(vaccination_percentages, results)

def run_super_spreaders(trials, filename):
    return run_trials(trials, filename, method="super_spreaders", vaccination_percentage=0)

def run_random_vaccination(trials, vaccination_percentage, filename):
    return run_trials(trials, filename, method="random_vaccination", vaccination_percentage=vaccination_percentage)

def run_trials(trials, filename, method, vaccination_percentage):
    n = 500
    transmission_chance, recovery_chance, connection_forming_chance = 0.20, 0.1, 0.03

    end_in_zero_count = 0
    stable_infections = []
    results = []

    trial_results = Parallel(n_jobs=-1)(delayed(run_single_trial)(
        trial, n, transmission_chance, recovery_chance, connection_forming_chance, method, vaccination_percentage
    ) for trial in tqdm(range(trials), desc="Running Trials"))

    for trial_result in trial_results:
        results.append(trial_result)
        if trial_result["end_in_zero"]:
            end_in_zero_count += 1
        else:
            stable_infections.append(apply_kalman_filter(trial_result["daily_infections"]))

    stable_avg = np.mean(stable_infections) if stable_infections else 0
    end_in_zero_percentage = end_in_zero_count / trials * 100

    save_results_to_csv(results, filename)

    return end_in_zero_percentage, results[0]['daily_infections'], stable_avg

def run_single_trial(trial, n, transmission_chance, recovery_chance, connection_forming_chance, method, vaccination_percentage):
    previous_status = np.zeros(n, dtype=int)
    connect_amount = np.zeros(n, dtype=int)
    super_spreaders = np.zeros(n, dtype=int)
    intercourse_chart = np.zeros((n, n), dtype=int)

    fill_intercourse_array(intercourse_chart, n, connection_forming_chance)
    fill_connect_amount(connect_amount, n, intercourse_chart)

    if method == "super_spreaders":
        fill_super_spreaders(super_spreaders, n, connect_amount)
        fill_status_array_with_super_spreaders(previous_status, n, super_spreaders)
    elif method == "random_vaccination":
        random_fill_status_array(previous_status, n, vaccination_percentage)

    infection_counts = run_simulation(transmission_chance, recovery_chance, intercourse_chart, previous_status, n)
    trial_result = {
        "trial": trial + 1,
        "end_in_zero": infection_counts[-1] == 0,
        "daily_infections": infection_counts
    }
    return trial_result

def run_simulation(transmission_chance, recovery_chance, intercourse_chart, previous_status, n):
    infection_counts = []
    max_day = 200
    current_status = previous_status.copy()

    for day in range(1, max_day + 1):
        all_recovered = True

        infected_indices = np.where(current_status == 1)[0]
        for i in infected_indices:
            for j in range(i):
                if intercourse_chart[i, j] == 1 and current_status[i] != current_status[j] and current_status[i] != -1 and current_status[j] != -1:
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

def apply_kalman_filter(data):
    n_iter = len(data)
    Q, R = 1e-5, 0.01
    xhat = np.zeros(n_iter)
    P = np.zeros(n_iter)
    xhatminus = np.zeros(n_iter)
    Pminus = np.zeros(n_iter)
    K = np.zeros(n_iter)
    xhat[0], P[0] = data[0], 1.0

    for k in range(1, n_iter):
        xhatminus[k], Pminus[k] = xhat[k-1], P[k-1] + Q
        K[k] = Pminus[k] / (Pminus[k] + R)
        xhat[k] = xhatminus[k] + K[k] * (data[k] - xhatminus[k])
        P[k] = (1 - K[k]) * Pminus[k]

    return np.mean(xhat[-10:])

def fill_intercourse_array(array, n, connection_forming_chance):
    for i in range(n):
        if i < 200:
            # Initially connect the first few nodes to each other to seed the network
            for j in range(i):
                array[i, j] = 1
        else:
            # Preferential attachment: nodes with more connections are more likely to receive new connections
            target_nodes = np.random.choice(range(i), size=int(connection_forming_chance * i), replace=True)
            for j in target_nodes:
                array[i, j] = 1
    for i in range(n):
        for j in range(i):
            array[j, i] = array[i, j]

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

def fill_connect_amount(array, n, intercourse_chart):
    array[:] = np.sum(intercourse_chart, axis=1) + np.sum(intercourse_chart, axis=0)

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

def plot_results(vaccination_percentages, results):
    percentages = [result[0] for result in results]
    end_in_zero = [result[1] for result in results]
    stable_avg = [result[2] for result in results]

    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(percentages, end_in_zero, marker='o')
    plt.xlabel('Vaccination Percentage')
    plt.ylabel('Percentage of Trials Ending in Zero Infections')
    plt.title('Zero Infection Trials vs. Vaccination Percentage')

    plt.subplot(1, 2, 2)
    plt.plot(percentages, stable_avg, marker='o')
    plt.xlabel('Vaccination Percentage')
    plt.ylabel('Average Number of Infected People (Stable Phase)')
    plt.title('Average Infected People vs. Vaccination Percentage')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
