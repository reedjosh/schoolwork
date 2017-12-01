#!/usr/bin/env python3
"""
Joshua Reed
parse_weather.py
Fall, 2017
A script to parse us monthly temperature averages and US monthly populations.
"""
def get_weather_data():
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
                    years[year][key].append(line[17+idx*9:22+idx*9])
   
        # Combine data from all weather stations into National Averages. 
        for year in years:
            for month in years[year]:
                # If data isn't known, -9999 was put in it's place.  
                # Filter -9999s.
                years[year][month] = [temp for temp in years[year][month] if not temp=="-9999"]
    
                total=0
                for temp in years[year][month]:
                    total+=int(temp)
                average = total/len(years["1988"][month])
                print(month, year)
                print(average)

def get_population_data():
    """Parses population data. TODO modify as needed when further specs are set.
    """
    with open('population.txt', 'r') as f:
        years = dict()
        for line in f:
            year = line[7:11]
            month = line[0:3]
            population = line[12:18]
            if year not in years: years[year]=dict()
            years[year][month]=population
            print(year)
            print(years[year])
        for year in years:
            print(year, years[year])

get_population_data()
