import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import pandas as pd

def pol2cart(rho, phi):
    x = rho * np.cos(np.pi*(phi/180))
    y = rho * np.sin(np.pi*(phi/180))
    return[x, y]


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


def wind_sampling(wind):

    divisions = 9

    cart_values = []
    pol_values = []
    for i in range(0, divisions):
        cart_values.append(pol2cart(wind, i*360/divisions))
        pol_values.append([wind, i*360/divisions])
    cart_values = np.array(cart_values)
    return cart_values


color=['blue','green']
for k in range(0,2):
    X = frame[frame["cluster"]==k]
    plt.scatter(X["Weight"],X["Height"],c=color[k])

color=['black','red','purple']
for j in range(6,24,6):
    wind = j

    cart_values = wind_sampling(wind)

    weights = gmm.predict_proba(cart_values)
    density = []
    for i in weights:
        density.append(max(i[0],i[1]))
    total = sum(density)
    for i in range(0,len(density)):
        density[i] = density[i]/total
        print(density[i], color[int(j/6 - 1)])

    for i in range(0,0):
        plt.scatter(cart_values[i][0],cart_values[i][1],c=color[int(j/6 - 1)])

plt.show()
