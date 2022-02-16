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
import matplotlib.pyplot as plt
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

def power(GMM, Turbines, wind_speed):
    """
    Uses wake_model.py which is used for caluclating the power output
    Inputs the GMM, a list for the wind farm layout and the wind speed
    Returns the expected power output of a wind farm layout
    """
    DIVISIONS = 72

    cart_values = []
    pol_values = []
    for i in range(0, DIVISIONS):
        cart_values.append(pol2cart(wind_speed, i*360/DIVISIONS))
        pol_values.append([wind_speed, i*360/DIVISIONS])
    cart_values = np.array(cart_values)

    weights = GMM.predict_proba(cart_values)
    density = []
    for i in weights:
        density.append(max(i[0],i[1]))
    total = sum(density)
    for i in range(0,len(density)):
        density[i] = density[i]/total

    AveragePowerOutput = 0
    for i in range(0, DIVISIONS):
        AveragePowerOutput += wake_model.Wake(Turbines, wind_speed, i*360/DIVISIONS)*density[i]
    AveragePowerOutput = AveragePowerOutput/DIVISIONS
    return AveragePowerOutput


dataset = []

num_lines = sum(1 for line in open('wind_data_combined.txt'))
f = open("wind_data_combined.txt", "r")
f.readline()
count=1
for i in range(num_lines):
    line = f.readline()
    try:
        line = line.split(",")
        line[0] = float(line[0])
        line[1] = float(line[1][:-1])
        dataset.append(pol2cart(line[0],line[1]))
    except:
        continue
    count+=1
f.close()

X = np.array(dataset)
gmm = GaussianMixture(n_components=2).fit(X)
labels = gmm.predict(X)
frame = pd.DataFrame(X)


frame['cluster'] = labels
frame.columns = ['Weight', 'Height', 'cluster']

wind = 10

class ProblemWrapper(Problem):

    def _evaluate(self, designs, out, *args, **kwargs):
        res = []
        for design in designs:
            res.append([0-power(gmm, design, wind), cost(design)])

        out['F'] = np.array(res)

problem = ProblemWrapper(n_var=400, n_obj=2, xl=[True]*400, xu=[False]*400)


algorithm = NSGA2(pop_size=5,
    sampling=get_sampling("bin_random"),
    crossover=get_crossover("bin_two_point"),
    mutation=get_mutation("bin_bitflip"),
    eliminate_duplicates=True)

stop_criteria = ('n_gen', 5)

results = minimize(
    problem=problem,
    algorithm=algorithm,
    termination=stop_criteria
)

res_data = results.F.T
fig = go.Figure(data=go.Scatter(x=0-res_data[0], y=res_data[1], mode='markers'))
fig.show()


'''
T = np.full((400), True)

Turbine_cost = cost(T)
print(Turbine_cost)
Power_Output = power(gmm, T, wind)
print(Power_Output)
'''

'''
color=['blue','green']
for k in range(0,2):
    X = frame[frame["cluster"]==k]
    plt.scatter(X["Weight"],X["Height"],c=color[k])
plt.show()
'''
