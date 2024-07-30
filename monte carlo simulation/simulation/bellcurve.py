import pandas as pd
import numpy as np

def main():
    n = 500
    connection_forming_chance = 0.008
    intercourse_chart = np.zeros((n, n), dtype=int)
    fill_intercourse_array(intercourse_chart, n, connection_forming_chance)
    
    connection_count = np.sum(intercourse_chart, axis=1)
    unique_values, counts = np.unique(connection_count, return_counts=True)
    
    results = []
    for value, count in zip(unique_values, counts):
        results.append((value, count))
    
    return results

def fill_intercourse_array(array, n, connection_forming_chance):
    random_matrix = np.random.rand(n, n)
    array[:] = (random_matrix < connection_forming_chance).astype(int)
    np.fill_diagonal(array, 0)

def run_simulations_and_save_to_csv(num_simulations, output_file):
    all_results = []
    for _ in range(num_simulations):
        results = main()
        all_results.extend(results)
    
    df = pd.DataFrame(all_results, columns=["Connections", "Count"])
    df.to_csv(output_file, index=False)

if __name__ == "__main__":
    run_simulations_and_save_to_csv(500, "simulation_results.csv")