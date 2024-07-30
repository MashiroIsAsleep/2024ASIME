import pandas as pd
import matplotlib.pyplot as plt
import statistics
import numpy as np
from sympy import symbols, Eq
from sympy.printing.latex import latex
from numpy.polynomial import Polynomial

def graphStdev(directory, hexcolor):
    data = pd.read_csv(directory)
    grouped_data = data.groupby('day')['infections']
    average = grouped_data.mean()
    stdev = grouped_data.std()
    plt.plot(average.index, average, label='Average Infections', color=hexcolor)
    plt.fill_between(average.index, average+stdev, average-stdev, color=hexcolor, alpha=0.1)
    
def graphWithoutStdev(directory, hexcolor):
    data = pd.read_csv(directory)
    grouped_data = data.groupby('day')['infections']
    average = grouped_data.mean()
    plt.plot(average.index, average, label='Average Infections', color=hexcolor)
    
def graphInitialize(x,y,title):
    plt.figure(figsize=(x, y))
    plt.title(title)
    plt.xlabel('Day')
    plt.ylabel('Infections')
    plt.grid(True)
    

def findMostSimilar(averagePartnerNumber):
    totalMeanDifference = np.zeros(29)
    # Read the super spreaders CSV file
    ssfilename = f"/kaggle/input/666weirdstuff/av{averagePartnerNumber}partner/super_spreaders.csv"
    ss = pd.read_csv(ssfilename)
    ssgrouped_data = ss.groupby('day')['infections']
    ssaverage = ssgrouped_data.mean()
    
    for i in range(0,29,1):
        index = i / 2
        # Dynamically generate the filename for random vaccination
        filename = f"/kaggle/input/666weirdstuff//av{averagePartnerNumber}partner/random_vaccination_{(index):.2f}.csv"
        rv = pd.read_csv(filename)
        rvgrouped_data = rv.groupby('day')['infections']
        rvaverage = rvgrouped_data.mean()
        # Calculate the mean difference and store it
        totalMeanDifference[i] = abs((ssaverage - rvaverage).sum())

    # Find the index of the smallest mean difference
    min_diff_index = np.argmin(totalMeanDifference)
    # Calculate the corresponding percentage for the smallest mean difference
    best_percentage = min_diff_index * 0.5

    return best_percentage
