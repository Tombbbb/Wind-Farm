""" Machine Learning to optimise wind turbine placements in a wind farm

Creates a Gaucian Mixture Model for existing wind data for a wind farm in La Haute Borne
Uses the wake model and the GMM to generate expected power output for a given wind farm layout
Uses Multi-Objective Optinisation to find pareto front of wind farm layouts
"""

from sklearn.mixture import GaussianMixture
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.visualization.scatter import Scatter
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import wake_model
import math

def pol2cart(rho, phi):
    """
    Converts polar coordinates to cartesian coordinates
    input is the wind speed and wind direction
    output is the north-south and east-west components of the wind
    """
    x = rho * np.cos(np.pi*(phi/180))
    y = rho * np.sin(np.pi*(phi/180))
    return[x, y]

def cost(L):
    '''
    The cost is a function of the number of wind turbines
    Inputs a list that represents the wind farm layout
    Returns the cost of the windfarm
    '''
    Turbine_count = 0
    for i in L:
        if i == True:
            Turbine_count += 1
    C = Turbine_count*((2/3) + (1/3)*(math.e**(-0.00174*((Turbine_count)**2))))
    return C

def power(GMM, Turbines, freq, max_speed):
    """
    Uses wake_model.py which is used for caluclating the power output
    Inputs the GMM, a list for the wind farm layout and the wind speed
    Returns the expected power output of a wind farm layout
    """
    DIR_DIVISIONS = 18

    wind_sum = sum(freq)
    power_sum = 0
    for i in range(len(freq)):
        wind_speed = (i+0.5)*max_speed/len(freq)
        cart_values = []
        pol_values = []
        for j in range(0, DIR_DIVISIONS):
            cart_values.append(pol2cart(wind_speed, j*360/DIR_DIVISIONS))
            pol_values.append([wind_speed, j*360/DIR_DIVISIONS])
        cart_values = np.array(cart_values)

        weights = GMM.predict_proba(cart_values)
        density = []
        for j in weights:
            density.append(max(j[0],j[1]))
        total = sum(density)
        for j in range(0,len(density)):
            density[j] = density[j]/total

        LocalAveragePowerOutput = 0
        for j in range(0, DIR_DIVISIONS):
            LocalAveragePowerOutput += wake_model.Wake(
                Turbines,
                wind_speed,
                j*360/DIR_DIVISIONS
            )*density[j]
        LocalAveragePowerOutput = LocalAveragePowerOutput/DIR_DIVISIONS
        power_sum += LocalAveragePowerOutput*freq[i]/wind_sum
    return power_sum


SPEED_DIVISIONS = 6
MAX_SPEED = 20
#the real maximun value is 21.23

dataset = []
speed_frequency = [0]*SPEED_DIVISIONS

num_lines = sum(1 for line in open('wind_data_combined.txt'))
g = open('results.txt', 'w')
g.close()
f = open("wind_data_combined.txt", "r")
f.readline()
for i in range(num_lines):
    line = f.readline()
    try:
        line = line.split(",")
        line[0] = float(line[0])
        line[1] = float(line[1][:-1])
        speed_frequency[int(round(SPEED_DIVISIONS*(line[0]/MAX_SPEED)))] += 1
        dataset.append(pol2cart(line[0],line[1]))
    except:
        continue
f.close()

print(speed_frequency)

X = np.array(dataset)
gmm = GaussianMixture(n_components=2).fit(X)
labels = gmm.predict(X)
frame = pd.DataFrame(X)


frame['cluster'] = labels
frame.columns = ['Weight', 'Height', 'cluster']

POPULATION = 3
GENERATIONS = 3

class ProblemWrapper(Problem):

    def _evaluate(self, designs, out, *args, **kwargs):
        res = []
        des = []
        for design in designs:
            c = cost(design)
            res.append([-power(gmm, design, speed_frequency, MAX_SPEED), c])
            des.append([design, c])

        out['F'] = np.array(res)
        out['G'] = des

problem = ProblemWrapper(n_var=400, n_obj=2, xl=[True]*400, xu=[False]*400)

algorithm = NSGA2(pop_size = POPULATION,
    sampling = get_sampling("bin_random"),
    crossover = get_crossover("bin_two_point"),
    mutation = get_mutation("bin_bitflip", prob = 0.05),
    eliminate_duplicates = True)

stop_criteria = ('n_gen', GENERATIONS)

results = minimize(
    problem = problem,
    algorithm = algorithm,
    termination = stop_criteria
)

res_data = results.F.T
fig = go.Figure(data=go.Scatter(x=-res_data[0], y=res_data[1], mode='markers'))

des_data = sorted(results.G, key=lambda x: x[1])
for i in des_data:
    print()
    wake_model.print_wind_farm(i[0])

fig.show()
