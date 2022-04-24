from utils import *
import matplotlib.pyplot as plt
from time import sleep
from SA import SA
import argparse

import timeit

start = timeit.default_timer()

parser = argparse.ArgumentParser()
# TODO: Force different seed from argparse if it exists against the one in the json file
parser.add_argument('--cooling', type=float, default=0.95)
parser.add_argument('--temp', type=int, default=1000)
args = parser.parse_args()



city_csv_file_path = "../city/city.csv"

visualize = True#False
visualization_rate = 0.001

initial_temp = args.temp #500
cooling_rate = args.cooling #0.999

csv_data = read_csv(city_csv_file_path)
cities = get_most_populated_cities(30, csv_data)
# print(cit)
cities = get_coord_cities(cities)
# print(cities)
# print("====================")
cities = get_xy_gps(cities)
# print(cities)
# print("====================")

cities = get_distances(cities)
# print(cities)
# print("====================")


fig, ax = None, None
if(visualize):
    plt.ion()
    plt.show(block=True) # block=True lets the window stay open at the end of the animation.
    fig, ax = plt.subplots()

# path = generate_path(to_list(cities))
# plot_lines(fig, ax, path)
# plt.show()
# exit()
# print(to_list(cities)[0]["distances"])

start = timeit.default_timer()
costs, temps, new_solution, new_solution_cost = SA(cities, initial_temp, cooling_rate, visualize=visualize, visualization_rate=visualization_rate, fig=fig, ax=ax)
stop = timeit.default_timer()

print('Time: ', stop - start)
print(f"Final Cost: {costs[-1]}")
print(f"Number of steps: {len(costs)}")

path = generate_path(new_solution)
title = f"Final solution, Cost={costs[-1]:.3f}\nInitial temp={initial_temp}, Cooling rate={cooling_rate}"
plot_animation(fig, ax, cities, path, pause=5, title=title)
clear_plot(ax)
plt.plot([i for i in range(len(costs))], [c/costs[0]for c in costs], label="Cost")
plt.legend()
plt.plot([i for i in range(len(temps))], [t/temps[0] for t in temps], label="Temp")
plt.legend()
plt.title(f"Initial temp={initial_temp}, Cooling rate={cooling_rate}")
plt.draw()
plt.pause(3)
# for i in range(100):
#     x = range(i)
#     y = range(i)
#     # plt.gca().cla() # optionally clear axes
#     plt.plot(x, y)
#     plt.title(str(i))
#     plt.draw()
#     plt.pause(0.1)



# ax = plot_cities(fig, ax, cities)
# # ax = plot_lines(fig, ax, cities)

# plt.draw()
# # sleep(5)
# plt.pause(1)
# clear_plot(ax)
# ax = plot_lines(fig, ax, cities)
# plt.draw()

# plt.pause(1)

