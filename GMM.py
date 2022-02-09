import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import pandas as pd
import wake_model
import math

def pol2cart(rho, phi):
    x = rho * np.cos(np.pi*(phi/180))
    y = rho * np.sin(np.pi*(phi/180))
    return[x, y]

def cost(L):
    Turbine_count = 0
    for i in L:
        if i == True:
            Turbine_count += 1
    C = Turbine_count*((2/3) + (1/3)*(math.e**(-0.00174*((Turbine_count)**2))))
    return C

def power(GMM, Turbines, wind_speed):
    divisions = 72

    cart_values = []
    pol_values = []
    for i in range(0, divisions):
        cart_values.append(pol2cart(wind_speed, i*360/divisions))
        pol_values.append([wind_speed, i*360/divisions])
    cart_values = np.array(cart_values)

    weights = GMM.predict_proba(cart_values)
    density = []
    for i in weights:
        density.append(max(i[0],i[1]))
    total = sum(density)
    for i in range(0,len(density)):
        density[i] = density[i]/total

    AveragePowerOutput = 0
    for i in range(0, divisions):
        AveragePowerOutput += wake_model.Wake(Turbines, wind_speed, i*360/divisions)*density[i]
    AveragePowerOutput = AveragePowerOutput/divisions
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


T = np.full((400), False)

Turbine_cost = cost(T)
print(Turbine_cost)
Power_Output = power(gmm, T, wind)
print(Power_Output)

'''
color=['blue','green']
for k in range(0,2):
    X = frame[frame["cluster"]==k]
    plt.scatter(X["Weight"],X["Height"],c=color[k])
plt.show()
'''
