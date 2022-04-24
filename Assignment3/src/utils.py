from audioop import reverse
import csv
from shapely.affinity import translate
from shapely.geometry import Point
import matplotlib.pyplot as plt
import numpy as np
from geopy import distance

# Source: https://realpython.com/python-csv/
def read_csv(filename):
    with open(filename, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        rows = []
        line_count = 0
        for row in csv_reader:
        #     if line_count == 0:
        #         print(f'Column names are {", ".join(row)}')
        #         line_count += 1
            rows.append(row)
        #     line_count += 1
        # print(f'Processed {line_count} lines.')
        return rows

# list_dict
def get_coord_cities(cities_list):
    cities = {}
    for e in cities_list:
        cities[e["address"]] = {"geo": [float(e["geo_lat"]), float(e["geo_lon"])]}
    return cities

def get_most_populated_cities(num, cities_list):
    cities = {}
    for e in cities_list:
        cities[e["address"]] = [int(e["population"]),e]
        # print(e["address"])
        # print(e)
        # print("-------------------------")
    # https://stackoverflow.com/questions/613183/how-do-i-sort-a-dictionary-by-value
    # cities = {"1":3, "2":1, "3":8, "4":5}
    sorted_cities = sorted(cities.items(), key=lambda x: x[1][0], reverse=True)
    # print(sorted_cities)
    filtered_cities = []
    for name,e in sorted_cities[:num]:
        # print(e)
        # print("---------------------------")
        filtered_cities.append(e[1])
        
    # print(filtered_cities)
    return filtered_cities

def get_xy_gps(cities):
    new_cities = {}
    for city_name in cities.keys():
        city = cities[city_name]
        geo = city["geo"]
        xy = translate(Point(geo))
        y,x = xy.x, xy.y
        xy = {"xy":[x,y]}
        geo = {"geo": geo}
        new_cities[city_name] = {**xy,**geo}
    return new_cities

def get_distances(cities):
    new_cities = {}
    for city_name in cities.keys():
        city = cities[city_name]
        # m = city["xy"]
        m = city["geo"]
        distances = {}
        for city_name2 in cities.keys():
            city2 = cities[city_name2]
            # m2 = city2["xy"]
            m2 = city2["geo"]
            if(city_name == city_name2):
                continue
            # city_distance = np.linalg.norm(np.array(m)-np.array(m2))
            city_distance = distance.distance(m, m2).km#/1000

            distances[city_name2] = city_distance
        new_cities[city_name] = {**city,**{"distances":distances}}
    return new_cities

# https://stackoverflow.com/questions/14432557/matplotlib-scatter-plot-with-different-text-at-each-data-point
def plot_cities(fig, ax, cities):
    x, y, n = [], [], []
    for city_name in cities.keys():
        city = cities[city_name]
        xy = city["xy"]
        x.append(xy[0])
        y.append(xy[1])
        n.append(city_name)
    ax.scatter(x, y)

    for i, txt in enumerate(n):
        ax.annotate(txt, (x[i], y[i]))
    return ax

def clear_plot(ax):
    ax.clear()

def plot_lines(fig, ax, cities):
    x, y, n = [], [], []

    for i in range(len(cities)):
        x.append(cities[i][0])
        y.append(cities[i][1])
        
    # for city_name in cities.keys():
    #     city = cities[city_name]
    #     xy = city["xy"]
    #     x.append(xy[0])
    #     y.append(xy[1])
    #     n.append(city_name)
    ax.plot(x, y, "-o")

    for i, txt in enumerate(n):
        ax.annotate(txt, (x[i], y[i]))
    return ax

def plot_animation(fig, ax, cities, lines, title=None, pause=0.1):
    clear_plot(ax)
    ax = plot_cities(fig, ax, cities)
    ax = plot_lines(fig, ax, lines)
    if(title is not None):
        ax.set_title(title)
    plt.draw()
    plt.pause(pause)
    
def to_list(cities):
    cities_list = []
    for city_name in cities.keys():
        city = cities[city_name]
        new_city = {**{"name":city_name}, **city}
        cities_list.append(new_city)
    return cities_list

def to_dict(cities_list):
    cities_dict = {}
    for i in range(len(cities_list)):
        cities_dict[cities_list[i]["name"]] = cities_list[i]
    return cities_dict

def generate_path(cities_list):
    xy = []
    for i in range(len(cities_list)):
        xy.append(cities_list[i]["xy"])
        # xy.append(cities_list[i]["xy"])
    # xy.append(cities_list[-1]["xy"])
    xy.append(cities_list[0]["xy"])
    return xy