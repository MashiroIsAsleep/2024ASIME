import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

def main():
    trials = 300  # Number of trials for better parallelism
    n = 500
    transmission_chance, recovery_chance, connection_forming_chance = 0.5, 0.15, 0.012
    immunity_duration = 30  # The duration for which vaccination remains effective
    revaccination_delay = 10  # Delay period before a person can be revaccinated

    # Create results directory if it does not exist
    os.makedirs("Results/waningImmunity/av4partner", exist_ok=True)
    
    # Run the simulation using the top 5% super spreaders method
    result = run_trials(trials, transmission_chance, recovery_chance, connection_forming_chance, immunity_duration, revaccination_delay, n, "Results/waningImmunity/av4partner/super_spreaders.csv", vaccinate_superspreaders=True)
    print(f"Super Spreaders Method:")
    print(f"Percentage of trials ending in zero infections: {result[0]}%")

    # Run the simulation using the random vaccination method
    for i in range(30):
        p = i * 0.005
        filename = f"Results/waningImmunity/av4partner/random_vaccination_{p*100:.2f}.csv"
        result = run_trials(trials, transmission_chance, recovery_chance, connection_forming_chance, immunity_duration, revaccination_delay, n, filename, vaccination_percentage=p)
        print(f"Random Vaccination Method for {p*100:.2f}% of the population:")
        print(f"Percentage of trials ending in zero infections: {result[0]}%")

def run_trials(trials, transmission_chance, recovery_chance, connection_forming_chance, immunity_duration, revaccination_delay, n, filename, vaccinate_superspreaders=False, vaccination_percentage=0):
    # Using joblib to parallelize the trials
    results = Parallel(n_jobs=-1)(delayed(run_single_trial)(trial_num, transmission_chance, recovery_chance, connection_forming_chance, immunity_duration, revaccination_delay, n, vaccinate_superspreaders, vaccination_percentage) for trial_num in tqdm(range(1, trials + 1), desc="Running Trials"))

    end_in_zero_count = sum(1 for result in results if result['end_in_zero'])
    end_in_zero_percentage = end_in_zero_count / trials * 100

    # Save the results to a CSV file
    save_results_to_csv(results, filename, immunity_duration)

    return end_in_zero_percentage, results[0]['daily_infections']

def run_single_trial(trial_num, transmission_chance, recovery_chance, connection_forming_chance, immunity_duration, revaccination_delay, n, vaccinate_superspreaders, vaccination_percentage):
    previous_status = np.zeros(n, dtype=int)
    connect_amount = np.zeros(n, dtype=int)
    super_spreaders = np.zeros(n, dtype=int)
    intercourse_chart = np.zeros((n, n), dtype=int)
    vaccinated_count = 0
    vaccination_counters = {}
    last_immunity_loss = {}

    fill_intercourse_array(intercourse_chart, n, connection_forming_chance)
    if vaccinate_superspreaders:
        connect_amount[:] = np.sum(intercourse_chart, axis=1) + np.sum(intercourse_chart, axis=0)
        fill_super_spreaders(super_spreaders, n, connect_amount)

    # Ensure at least one person is initially infected
    initial_infected = select_initial_infected(n, super_spreaders if vaccinate_superspreaders else None)
    previous_status[initial_infected] = 1

    infection_counts, daily_vaccinations, vaccination_counters = run_simulation(
        transmission_chance, recovery_chance, intercourse_chart, previous_status, super_spreaders,
        vaccinated_count, vaccination_counters, last_immunity_loss, immunity_duration, revaccination_delay,
        n, vaccinate_superspreaders, vaccination_percentage
    )
    trial_result = {
        "trial": trial_num,
        "end_in_zero": infection_counts[-1] == 0,
        "daily_infections": infection_counts,
        "daily_vaccinations": daily_vaccinations,
        "vaccination_counters": vaccination_counters
    }
    return trial_result

def run_simulation(transmission_chance, recovery_chance, intercourse_chart, previous_status, super_spreaders, vaccinated_count, vaccination_counters, last_immunity_loss, immunity_duration, revaccination_delay, n, vaccinate_superspreaders, vaccination_percentage=0):
    infection_counts = []
    daily_vaccinations = []
    max_day = 200

    for day in range(1, max_day + 1):
        current_status = previous_status.copy()

        # Daily vaccination process
        max_vaccinations = int(n * 0.05) if vaccinate_superspreaders else int(n * vaccination_percentage)
        remaining_vaccinations = max_vaccinations - vaccinated_count
        if remaining_vaccinations > 0:
            choices = np.arange(remaining_vaccinations + 1)
            # Adjust weights for slower vaccination process
            weights = np.exp(-0.5 * choices)  # Exponential decay
            weights = weights / weights.sum()

            rng_vaccination_count = np.random.choice(choices, p=weights)
        else:
            rng_vaccination_count = 0

        if rng_vaccination_count > 0:
            if vaccinate_superspreaders:
                super_spreader_indices = np.where(super_spreaders == 1)[0]
                potential_vaccinees = np.setdiff1d(super_spreader_indices, np.where(previous_status == -1)[0])
            else:
                potential_vaccinees = np.setdiff1d(np.arange(n), np.where(previous_status == -1)[0])
            
            if len(potential_vaccinees) > 0:
                # Filter out individuals who lost immunity within the last 10 days
                potential_vaccinees = [idx for idx in potential_vaccinees if idx not in last_immunity_loss or (day - last_immunity_loss[idx]) > revaccination_delay]
                if len(potential_vaccinees) > 0:
                    new_vaccinees = np.random.choice(potential_vaccinees, min(rng_vaccination_count, len(potential_vaccinees)), replace=False)
                    previous_status[new_vaccinees] = -1
                    for idx in new_vaccinees:
                        vaccination_counters[idx] = 0
                    vaccinated_count += len(new_vaccinees)
                    daily_vaccinations.append(len(new_vaccinees))
                else:
                    daily_vaccinations.append(0)
            else:
                daily_vaccinations.append(0)
        else:
            daily_vaccinations.append(0)

        # Increment vaccination counters and check for expiring immunity
        for idx, count in list(vaccination_counters.items()):
            vaccination_counters[idx] += 1
            if vaccination_counters[idx] >= immunity_duration:
                previous_status[idx] = 0  # Update status to 0
                last_immunity_loss[idx] = day
                del vaccination_counters[idx]
                vaccinated_count -= 1  # Decrement the total vaccinated count
                print(f"Day {day}: Immunity waned for individual {idx}, updating status to 0")

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
            daily_vaccinations.extend([0] * (max_day - day))
            break

        previous_status[:] = current_status
    return infection_counts, daily_vaccinations, vaccination_counters

def fill_intercourse_array(array, n, connection_forming_chance):
    random_matrix = np.random.rand(n, n)
    array[:] = (random_matrix < connection_forming_chance).astype(int)
    np.fill_diagonal(array, 0)

def select_initial_infected(n, super_spreaders=None):
    if super_spreaders is not None:
        non_super_spreaders_indices = np.where(super_spreaders == 0)[0]
        return np.random.choice(non_super_spreaders_indices, 1)
    return np.random.choice(np.arange(n), 1)

def fill_super_spreaders(array, n, connect_amount):
    super_spreaders_count = round(0.05 * n)
    top_indices = np.argpartition(connect_amount, -super_spreaders_count)[-super_spreaders_count:]
    array.fill(0)
    array[top_indices] = 1
   
def save_results_to_csv(results, filename, immunity_duration):
    data = []
    for result in results:
        daily_vaccinated = 0
        total_vaccinated = 0
        for day, (infections, vaccinated) in enumerate(zip(result['daily_infections'], result['daily_vaccinations'])):
            daily_vaccinated += vaccinated
            total_vaccinated = daily_vaccinated - sum(1 for count in result['vaccination_counters'].values() if count >= immunity_duration)
            data.append({
                "trial": result["trial"],
                "day": day + 1,
                "infections": infections,
                "vaccinated": total_vaccinated
            })
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)

if __name__ == "__main__":
    main()
