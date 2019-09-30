# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 17:36:07 2018

@author: KG
"""
import csv
from tmdbv3api import TMDb, Movie
import pandas as pd
tmdb = TMDb()
tmdb.api_key = 'd8abe6d21efb94f755ca5207b3aff3c7'
pre = "https://image.tmdb.org/t/p/w185_and_h278_bestv2"

test = pd.read_csv("C:/Users/akshatb/Desktop/BE-Project/Own Scrapper/IDset.csv")
ids = test["ID"]

for i in range(0,len(ids)):
    print(ids[i])

    movie = Movie()
    m = movie.details(ids[i])
    if m.title == None or len(m.genres) == 0 or m.poster_path == None:
        continue
    g = ""
    lis = []
    lis.append(pre+m.poster_path)
    lis.append(m.title)
    for genre in m.genres:
        g += " "
        g += genre['name']
        g += ','
    g = g.strip()
    g = g[:-1]
    lis.append(g)

    print(lis)
    with open('C:/Users/akshatb/Desktop/BE-Project/Own Scrapper/temp.csv', 'a') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(lis)
