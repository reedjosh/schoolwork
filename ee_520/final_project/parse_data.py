#!/usr/bin/env python3
"""
Joshua Reed
parse_weather.py
Fall, 2017
A script to parse us monthly temperature averages and US monthly populations.
"""
import matplotlib
import errno

try:
    matplotlib.use('agg')
except:
    # Some IDE, most likely
    pass

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy import stats
from math import exp as e
import seaborn as sns

def make_scatterplot_ex():
    df = sns.load_dataset('iris')
    sns.regplot(x=df["sepal_length"], y=df["sepal_width"])
    plt.savefig('img/scatter_ex.png')

def safe_make(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def get_temperature_data():
    """ Parses weather data. TODO modify as needed when further specs are set.
    """
    with open('weather.txt', 'r') as f:
        years = dict()
        for line in f:
            year = line[12:16]
            if int(year) >= 1900:
                if not year in years:
                    years[year]={"Jan":[], "Feb":[], "Mar":[], "Apr":[], "May":[], 
                                 "Jun":[], "Jul":[], "Aug":[], "Sep":[], "Oct":[], 
                                 "Nov":[], "Dec":[]}
                for idx, key in enumerate(years[year]):
                    temp = line[17+idx*9:22+idx*9]
                    if  temp != '-9999': # Filter missing datapoints.
                        years[year][key].append(float(line[17+idx*9:22+idx*9])/10)
   
        # Combine data from all weather stations into National Averages. 
        for year in years:
            for month in years[year]:
                total=0
                for temp in years[year][month]:
                    total+=int(temp)
                average = total/len(years[year][month])
                years[year][month]=average
    return years

def get_population_data():
    """Parses population data. TODO modify as needed when further specs are set.
    """
    # Read in population data.
    # Will include year, month, and population in the form of:
    # years[year][month]=population
    years = dict()
    with open('population.txt', 'r') as f:
        for line in f:
            year = line[7:11]
            month = line[0:3]
            population = float(line[12:18])
            if year not in years: years[year]=dict()
            years[year][month]=population

    # For years before 1990, there is only yearly data. 
    # Here I have scipy interpolate the data.
    # Start by getting all of the 90 years of population data as a list.
    # Create the interpolation function. 
    # Put in 12 times the original X points.
    # Build back the data while keeping the original data points for July.
    populations = []
    for year in range(1900, 1990):
        populations.append(years[str(year)][month])
    months = len(populations)*12    
    populations = np.array(populations)
    x = np.arange(0, 90, 1)
    tck = interpolate.splrep(x, populations, s=0)
    xnew = np.arange(0, 90, 90/months) 
    ynew = interpolate.splev(xnew, tck, der=0)
    
    # Insert newly interpolated data into overall dataset.
    months=["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    for idx, population in enumerate(ynew, start=1):
        month_num = idx % 12
        year_num = 1900 + int(idx/12)
        month = months[month_num]
        if month_num != 11: # don't overwrite july datapoints
            years[str(year_num)][month]=population
    return years

def pop_to_growth_data(pop_data):
    previous_pop=1
    month_keys = ["Jan", "Feb", "Mar", "Apr", "May", 
                  "Jun", "Jul", "Aug", "Sep", "Oct", 
                  "Nov", "Dec"]
    for year in range(1991, 2015):
        year=str(year)
        for month in month_keys:
            pop = pop_data[year][month]
            pop_data[year][month]=(pop-previous_pop)/previous_pop*100
            previous_pop = pop
    return pop_data
        
        
def make_basic_scatter_temperature(temperature_data):
    """Create a scatter plot of temperature by month from 1990
    """
    data = get_temperature_data()    
    temps = []
    for year in range(1990, 2015):
        year = str(year)
        for month in data[year]:
            temps.append(temperature_data[year][month])
    y = np.array(temps)
    x = np.arange(0, y.size, 1)
    plt.legend()
    sns.regplot(x=x, y=y, fit_reg=False)
    plt.xlabel("Months since Jun., 1990")
    plt.rc('text', usetex=True)
    plt.ylabel(r"Temperature, $^\circ F$")
    plt.savefig('img/temperature_scatter_plot.png')
    plt.clf()


def make_monthly_scatter_temperature(temperature_data):
    """Create a scatter plot of temperature by month from 1990.
       Also color code and add a legend per-month.
    """
    monthly_temps = {"Jan":[], "Feb":[], "Mar":[], "Apr":[], "May":[], 
                     "Jun":[], "Jul":[], "Aug":[], "Sep":[], "Oct":[], 
                     "Nov":[], "Dec":[]}
    for year in range(1990, 2015):
        year = str(year)
        for month in temperature_data[year]:
            monthly_temps[month].append(temperature_data[year][month])
    for month in monthly_temps:
        y = np.array(monthly_temps[month])
        x = np.arange(0, y.size*12, 12)
        sns.regplot(x=x, y=y, fit_reg=False, label=month)
    plt.xlabel("Months since Jun., 1990")
    plt.rc('text', usetex=True)
    plt.ylabel(r"Temperature, $^\circ F$")
    ax=plt.subplot(111)
    chartBox = ax.get_position()
    ax.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.95, chartBox.height])
    ax.legend(loc='upper center', bbox_to_anchor=(1.08, 0.8), shadow=True, ncol=1)
    plt.savefig('img/monthly_temperature_scatter_plot.png')
    plt.clf()

def make_monthly_scatter_population(population_data):
    """Create a scatter plot of population by month from 1990.
       Also color code and add a legend per-month.
    """
    monthlypopulations = {"Jan":[], "Feb":[], "Mar":[], "Apr":[], "May":[], 
                     "Jun":[], "Jul":[], "Aug":[], "Sep":[], "Oct":[], 
                     "Nov":[], "Dec":[]}
    for year in range(1990, 2015):
        year = str(year)
        for month in population_data[year]:
            monthlypopulations[month].append(population_data[year][month])
    for month in monthlypopulations:
        y = np.array(monthlypopulations[month])
        x = np.arange(0, y.size*12, 12)
        sns.regplot(x=x, y=y, fit_reg=False, label=month)
    plt.xlabel("Months since Jun., 1990")
    plt.rc('text', usetex=True)
    plt.ylabel(r"Population, Millions")
    plt.title("US Monthly Population")
    ax=plt.subplot(111)
    chartBox = ax.get_position()
    ax.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.95, chartBox.height])
    ax.legend(loc='upper center', bbox_to_anchor=(1.08, 0.8), shadow=True, ncol=1)
    plt.savefig('img/monthly_population_scatter_plot.png')
    plt.clf()

def make_monthly_scatter_population_growth(population_data):
    """Create a scatter plot of population growth by month from 1990.
       Also color code and add a legend per-month.
    """
    monthlypopulations = {"Jan":[], "Feb":[], "Mar":[], "Apr":[], "May":[], 
                     "Jun":[], "Jul":[], "Aug":[], "Sep":[], "Oct":[], 
                     "Nov":[], "Dec":[]}
    growth_data = pop_to_growth_data(population_data)
    for year in range(1992, 2015):
        year = str(year)
        for month in growth_data[year]:
            monthlypopulations[month].append(growth_data[year][month])
    for month in monthlypopulations:
        y = np.array(monthlypopulations[month])
        x = np.arange(0, y.size*12, 12)
        sns.regplot(x=x, y=y, fit_reg=False, label=month)
    plt.xlabel("Months since Jun., 1990")
    plt.rc('text', usetex=True)
    plt.ylabel(r"Population Growth %")
    plt.title("US Monthly Population Growth")
    ax=plt.subplot(111)
    chartBox = ax.get_position()
    ax.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.95, chartBox.height])
    ax.legend(loc='upper center', bbox_to_anchor=(1.08, 0.8), shadow=True, ncol=1)
    plt.savefig('img/monthly_population_growth_scatter_plot.png')
    plt.clf()

def plot_monthly_temp_means(data):
    monthlypopulations = {"Jan":[], "Feb":[], "Mar":[], "Apr":[], "May":[], 
                     "Jun":[], "Jul":[], "Aug":[], "Sep":[], "Oct":[], 
                     "Nov":[], "Dec":[]}
    for year in range(1992, 2015):
        year = str(year)
        for month in data[year]:
            monthlypopulations[month].append(data[year][month])
    x = []
    y = []
    for month in monthlypopulations:
        y.append(np.mean(monthlypopulations[month]))
        x.append(month)
    y = np.array(y)
    print(x, y)
    sns.barplot(x=x, y=y)
    plt.savefig('img/monthly_temp_growth_means.png')
    plt.clf()

def plot_monthly_pop_means(data):
    monthlypopulations = {"Jan":[], "Feb":[], "Mar":[], "Apr":[], "May":[], 
                     "Jun":[], "Jul":[], "Aug":[], "Sep":[], "Oct":[], 
                     "Nov":[], "Dec":[]}
    for year in range(1992, 2015):
        year = str(year)
        for month in data[year]:
            monthlypopulations[month].append(data[year][month])
    x = []
    y = []
    for month in monthlypopulations:
        y.append(np.mean(monthlypopulations[month]))
        x.append(month)
    y = np.array(y)
    print(x, y)
    sns.barplot(x=x, y=y)
    plt.savefig('img/monthly_pop_growth_means.png')
    plt.clf()

def main():
    safe_make('img')
    matplotlib.rcParams['savefig.dpi'] = 150
    sns.set_style('whitegrid')

    temp_data = get_temperature_data()    
    #make_basic_scatter_temperature(temp_data)    
    #make_monthly_scatter_temperature(data)
    pop_data = get_population_data()    
    #make_monthly_scatter_population(pop_data)
    #make_monthly_scatter_population_growth(pop_data)
    growth_data = pop_to_growth_data(pop_data)
    plot_monthly_pop_means(growth_data)
    plot_monthly_temp_means(temp_data)
    

if __name__ == '__main__':
    main()



