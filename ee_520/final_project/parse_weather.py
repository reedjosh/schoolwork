#!/usr/bin/env python3
"""
Joshua Reed
parse_weather.py
Fall, 2017
A script to parse us monthly temperature averages.
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
