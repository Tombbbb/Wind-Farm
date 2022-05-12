"""
Machine Learning to optimise wind turbine placements in a wind farm

Creates a Gaucian Mixture Model for existing wind data for a wind farm in La Haute Borne
Uses the wake model and the GMM to generate expected power output for a given wind farm layout
Uses Multi-Objective Optinisation to find pareto front of wind farm layouts
"""

from sklearn.mixture import GaussianMixture
from pymoo.factory import get_sampling, get_crossover, get_mutation, get_reference_directions
from pymoo.visualization.scatter import Scatter
from pymoo.algorithms.moo.age import AGEMOEA
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.indicators.hv import Hypervolume
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import wake_model_compare_moo
import math

h = open('README.md', 'r') #Constants are found and can be changed in README.md
for i in range(3):
    h.readline()

DIR_DIVISIONS = int(str(h.readline())[16:-1]) #the number of samples for wind direction
SPEED_DIVISIONS = int(str(h.readline())[18:-1]) #the number of samples for wind speed
MAX_SPEED = int(str(h.readline())[12:-1]) #the real maximun value is 21.23
GENERATIONS = int(str(h.readline())[13:-1]) #the number of generations in the optimisation algorithm
POPULATION = int(str(h.readline())[13:-1]) #the population size of the optimisation algorithm
GRID = str(h.readline())[12:-1]
GRID_X, GRID_Y = "", "" #the size of the grid of potential wind turbines placements
x_check = False
for i in range(len(GRID)):
    if GRID[i] == "x":
        x_check = True
    elif x_check == False:
        GRID_X += str(GRID[i])
    elif x_check == True:
        GRID_Y += str(GRID[i])
GRID_X, GRID_Y = int(GRID_X), int(GRID_Y)
h.close()

def pol2cart(rho, phi):
    """
    Converts polar coordinates to cartesian coordinates
    input is the wind speed and wind direction
    output is the north-south and east-west components of the wind
    """
    x = rho * np.cos(np.pi*(phi/180)) #convert to radians
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
        if i == True: #counting the number of turbines
            Turbine_count += 1
    C = Turbine_count*((2/3) + (1/3)*(math.e**(-0.00174*((Turbine_count)**2))))
    return C #returns the cost of the wind farm

def power(GMM, Turbines, freq, max_speed, divs, GRID_X, GRID_Y):
    """
    Uses wake_model.py which is used for caluclating the power output
    Inputs the GMM, a list for the wind farm layout and the wind speed
    Returns the expected power output of a wind farm layout
    """
    DIVISIONS = divs #the number of divisions for wind direction

    wind_sum = sum(freq) #the total wind speed for all wind samples
    power_sum = 0 #the total power output
    for i in range(len(freq)):
        wind_speed = (i+0.5)*max_speed/len(freq) #wind speed for the wind divison
        cart_values = [] #list of cartesian coordinates
        pol_values = [] #list of polar coodrinates
        for j in range(0, DIVISIONS):
            cart_values.append(pol2cart(wind_speed, j*360/DIVISIONS))
            pol_values.append([wind_speed, j*360/DIVISIONS])
        cart_values = np.array(cart_values)

        #the weightings of the wind directions based on the gauccian mixture model
        weights = GMM.predict_proba(cart_values)
        density = [] #the density of wdind conditions within the gauccian mixture model
        for j in weights:
            density.append(max(j[0],j[1]))
        total = sum(density)

        LocalAveragePowerOutput = 0
        for j in range(0, DIVISIONS):
            LocalAveragePowerOutput += wake_model_compare_moo.Wake( #calculate power output
                Turbines,                               #for distinct wind speed
                wind_speed,                             #and wind direction
                j*360/DIVISIONS,
                GRID_X,
                GRID_Y
            )*density[j]/total
        power_sum += LocalAveragePowerOutput*freq[i]/(wind_sum*DIVISIONS)

    return power_sum


dataset = []
speed_frequency = [0]*SPEED_DIVISIONS

g = open('results_nsga2.txt', 'w') #clear text file
g.close()
g = open('results_nsga3.txt', 'w') #clear text file
g.close()
g = open('results_agemoea.txt', 'w') #clear text file
g.close()

num_lines = sum(1 for line in open('wind_data.txt'))
f = open("wind_data.txt", "r")
f.readline()
for i in range(num_lines):
    line = f.readline()
    try:
        line = line.split(",")
        line[0] = float(line[0]) #wind speed
        line[1] = float(line[1][:-1]) #wind direction, removes new line character
        speed_frequency[int(round(SPEED_DIVISIONS*(line[0]/MAX_SPEED)))] += 1
        dataset.append(pol2cart(line[0],line[1]))
    except:
        continue
f.close()

print(speed_frequency)

X = np.array(dataset)
gmm = GaussianMixture(n_components=2).fit(X) #generates gauccian mixture model
labels = gmm.predict(X)
frame = pd.DataFrame(X)


frame['cluster'] = labels
frame.columns = ['Weight', 'Height', 'cluster']

class ProblemWrapper(Problem):
    """
    Defines an instance of the wind farm
    """
    def _evaluate(self, designs, out, *args, **kwargs):
        res = []
        des = []
        for design in designs:
            c = cost(design)
            res.append([-power(gmm, design, speed_frequency, MAX_SPEED, DIR_DIVISIONS, GRID_X, GRID_Y), c])
            des.append([design, c])

        out['F'] = np.array(res)
        out['G'] = np.array(des)

problem = ProblemWrapper(n_var=GRID_X*GRID_Y, n_obj=2, xl=[True]*GRID_X*GRID_Y, xu=[False]*GRID_X*GRID_Y)
ref_dirs = get_reference_directions("energy", 2, GRID_X*GRID_Y)


algorithm1 = NSGA2(pop_size = POPULATION,
    ref_dirs = ref_dirs,
    sampling = get_sampling("bin_random"), #random binary sampling
    crossover = get_crossover("bin_two_point"), #binary two point crossover
    mutation = get_mutation("bin_bitflip", prob = 0.05), #binary bitflip mutation
    eliminate_duplicates = True)

algorithm2 = NSGA3(pop_size = POPULATION,
    ref_dirs = ref_dirs,
    sampling = get_sampling("bin_random"), #random binary sampling
    crossover = get_crossover("bin_two_point"), #binary two point crossover
    mutation = get_mutation("bin_bitflip", prob = 0.05), #binary bitflip mutation
    eliminate_duplicates = True)

algorithm3 = AGEMOEA(pop_size = POPULATION,
    ref_dirs = ref_dirs,
    sampling = get_sampling("bin_random"), #random binary sampling
    crossover = get_crossover("bin_two_point"), #binary two point crossover
    mutation = get_mutation("bin_bitflip", prob = 0.05), #binary bitflip mutation
    eliminate_duplicates = True)


stop_criteria = ('n_gen', GENERATIONS) #stops NSGA3 after n generations

algors = [algorithm1, algorithm2, algorithm3]
colours = ['black', 'red', 'blue']
plt.figure(figsize=(7, 5))

for j in range(3):
    results = minimize(
        problem = problem,
        algorithm = algors[j],
        termination = stop_criteria,
        save_history=True
    )

    res_data = results.F.T
    fig = go.Figure(data=go.Scatter(x=-res_data[0], y=res_data[1], mode='markers'))

    print(res_data)

    des_data = sorted(results.G, key=lambda x: x[1])
    for i in des_data: #itterate through pareto solutions
        wake_model_compare_moo.print_wind_farm(i[0], GRID_X, GRID_Y, j) #shows results in results.py

    fig.show()

    """
    Copied Hypervolume code
    """

    X, F = results.opt.get("X", "F")
    hist = results.history

    n_evals = []             # corresponding number of function evaluations\
    hist_F = []              # the objective space values in each generation
    hist_cv = []             # constraint violation in each generation
    hist_cv_avg = []         # average constraint violation in the whole population

    for algo in hist:

        # store the number of function evaluations
        n_evals.append(algo.evaluator.n_eval)

        # retrieve the optimum from the algorithm
        opt = algo.opt

        # store the least contraint violation and the average in each population
        hist_cv.append(opt.get("CV").min())
        hist_cv_avg.append(algo.pop.get("CV").mean())

        # filter out only the feasible and append and objective space values
        feas = np.where(opt.get("feasible"))[0]
        hist_F.append(opt.get("F")[feas])


    approx_ideal = F.min(axis=0)
    approx_nadir = F.max(axis=0)

    metric = Hypervolume(ref_point= np.array([1.1, 1.1]),
                         norm_ref_point=False,
                         zero_to_one=True,
                         ideal=approx_ideal,
                         nadir=approx_nadir)

    hv = [metric.do(_F) for _F in hist_F]

    if j == 0: #nsga2
        plt.plot(n_evals, hv,  color='black', lw=0.7, label="Avg. CV of Pop")
        plt.scatter(n_evals, hv,  facecolor="none", edgecolor='black', marker="p")
    elif j == 1: #nsga3
        plt.plot(n_evals, hv,  color='blue', lw=0.7, label="Avg. CV of Pop")
        plt.scatter(n_evals, hv,  facecolor="none", edgecolor='blue', marker="p")
    elif j == 2: #AGE/MOEA
        plt.plot(n_evals, hv,  color='green', lw=0.7, label="Avg. CV of Pop")
        plt.scatter(n_evals, hv,  facecolor="none", edgecolor='green', marker="p")
plt.title("Convergence")
plt.xlabel("Function Evaluations")
plt.ylabel("Hypervolume")
plt.show()
