"""
Contains funtions used for the wake model, power output and wind farm layout
"""

import math

class Turbines:
    def __init__(self):
        self.Positions = []
        self.r = 41  #rotor radius
        self.h_height = 80    #hub height

    def addTurbine(self, x, y):
        self.Positions.append([x,y])

    def getTurbines(self):
        return self.Positions

    def getRadius(self):
        return self.r


def getAngle(x, y, z):
    """
    Inputs 3 points
    returns the angle created between the 3 points
    """
    angle = math.degrees(math.atan2(z[1]-y[1], z[0]-y[0]) - math.atan2(x[1]-y[1], x[0]-y[0]))
    if angle < 0:
        angle += 360
    return angle

def Wake(turbine_string, wind_speed, wind_direcetion, x_coords, y_coords):
    """
    Inputs a list for the tubines, the wind speed and the wind direction
    Returns the expected power output of the wind farm layout using the wake model
    """
    Ct = 0.75 #Thrust Coefficient
    Kw = 0.038 #Wake Decay Coefficient
    Cp = 0.44 #power Coefficient
    p = 1.225 #air density
    min_TD = 246 #Minimum distance between turbines

    T = Turbines()
    for i in range(x_coords*y_coords):
        if turbine_string[i] == True:
            T.addTurbine(i % y_coords, math.floor(i/x_coords)) #x and y coordinates of the wind farm

    turbine_list = T.getTurbines()

    Turbine_power = 0
    for i in turbine_list:
        Turbine_A = i
        Turbine_wake = 0
        Turbine_wake_list = []
        for j in turbine_list:
            if i != j: #make sure the turbines are not the same turbine
                Turbine_B = j
                Turbine_angle = getAngle([Turbine_B[0]+1,Turbine_B[1]], Turbine_B, Turbine_A)
                wake_range = math.degrees(math.atan(Kw))
                if wind_direcetion + wake_range >= Turbine_angle and wind_direcetion - wake_range <= Turbine_angle:
                    diff = math.sqrt(((Turbine_A[0]-Turbine_B[0])*min_TD)**2 + ((Turbine_A[1]-Turbine_B[1])*min_TD)**2)
                    x = diff*math.cos(math.radians(Turbine_angle - wind_direcetion))
                    SD = (1-math.sqrt(1-Ct))/((1+(Kw*x)/(T.getRadius()))**2)
                    Turbine_wake_list.append(SD)
        if len(Turbine_wake_list) == 0: #if there is no wake effect from other turbines
            Turbine_wake = wind_speed
        else:
            for k in Turbine_wake_list:
                 Turbine_wake += k**2
            Turbine_wake = wind_speed*(1 - math.sqrt(Turbine_wake))
        Turbine_power += 0.5*p*math.pi*Cp*(T.getRadius()**2)*(Turbine_wake**3)

    return Turbine_power


def print_wind_farm(turbine_string, max_x, max_y, moo):
    """
    Inputs a list the represents the wind farm layout
    Prints the wind farm with an O where there is a turbine
    """
    wind_farm = []
    for i in range(0,max_y):
        wind_farm.append([None]*max_x) #empty wind farm array

    T = Turbines()
    for i in range(max_x*max_y):
        if turbine_string[i] == True:
            T.addTurbine(i % max_y, math.floor(i/max_x))

    turbine_list = T.getTurbines()
    for i in turbine_list:
        coords = i
        wind_farm[coords[0]][coords[1]] = i

    if moo == 0:
        f = open('results_nsga2.txt', 'a')
    if moo == 1:
        f = open('results_nsga3.txt', 'a')
    if moo == 2:
        f = open('results_agemoea.txt', 'a')

    for i in wind_farm:
        row = ""
        for j in i:
            if j != None:
                row += "O"
            else:
                row += "."
        f.write(row)
        f.write('\n')
    f.write('\n')

    f.close()

    return
