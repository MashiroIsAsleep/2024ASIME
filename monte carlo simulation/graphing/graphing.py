import pandas as pd
import matplotlib.pyplot as plt
import statistics

data = pd.read_csv('/file/directory/of/first/csv')
datai = pd.read_csv('/file/directory/of/second/csv')


# Calculate the average and standard deviation for each day
grouped_data = data.groupby('day')['infections']
average = grouped_data.mean()
stdev = grouped_data.std()

grouped_datai = datai.groupby('day ')['infections']
averagei = grouped_datai.mean()
stdevi = grouped_datai.std()

# Plot the results
plt.figure(figsize=(12, 6))
plt.title('Monte Carlo Simulation of Infections Over Days')
plt.xlabel('Day')
plt.ylabel('Infections')
plt.grid(True)

# Plotting the average infections as a bar chart
plt.plot(average.index, average, label='Average Infections', color='blue')
plt.fill_between(average.index, average+stdev, average-stdev, color='blue', alpha=0.1)

plt.plot(averagei.index, averagei, label='Average Infections', color='red')
plt.fill_between(averagei.index, averagei+stdevi, averagei-stdevi, color='red', alpha=0.1)

plt.show()