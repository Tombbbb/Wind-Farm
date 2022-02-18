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

def getEmptyWindFarmList():
    """
    returns an array for the wind farm layout
    """
    wind_farm = []
    for i in range(0,20):
        wind_farm.append([None]*20)
    return wind_farm


def Wake(turbine_string, wind_speed, wind_direcetion):
    """
    Inputs a list for the tubines, the wind speed and the wind direction
    Returns the expected power output of the wind farm layout using the wake model
    """
    Ct = 0.8 #Thrust Coefficient
    Kw = 0.2 #Wake Decay Coefficient
    a = 103.33
    m = 20.53
    n = 1190.73
    tau = 1.14
    VP = [a,m,n,tau] #vector parameter of the logistic function

    wind_farm = getEmptyWindFarmList()

    T = Turbines()
    for i in range(400):
        if turbine_string[i] == True:
            T.addTurbine(math.floor(i/20), i % 20)

    turbine_list = T.getTurbines()

    for i in turbine_list:
        coords = i
        wind_farm[coords[0]][coords[1]] = i

    Turbine_power = 0
    for i in turbine_list:
        Turbine_A = i
        Turbine_wake = 0
        Turbine_wake_list = []
        for j in turbine_list:
            if i != j:
                Turbine_B = j
                Turbine_angle = getAngle([Turbine_B[0]+1,Turbine_B[1]], Turbine_B, Turbine_A)
                wake_range = math.degrees(math.atan(Kw))
                if wind_direcetion + wake_range >= Turbine_angle and wind_direcetion - wake_range <= Turbine_angle:
                    diff = math.sqrt(((Turbine_A[0]-Turbine_B[0])*246)**2 + ((Turbine_A[1]-Turbine_B[1])*246)**2)
                    x = diff*math.cos(math.radians(Turbine_angle - wind_direcetion))
                    SD = (1-math.sqrt(1-Ct))/((1+(Kw*x)/(T.getRadius()))**2)
                    Turbine_wake_list.append(SD)
        if len(Turbine_wake_list) == 0:
            Turbine_wake = wind_speed
        else:
            for k in Turbine_wake_list:
                 Turbine_wake += k**2
            Turbine_wake = wind_speed*(1 - math.sqrt(Turbine_wake))
        Turbine_power += a*(1+m*(math.e**(-(Turbine_wake/(T.getRadius())))))/(1+n*(math.e**(-(Turbine_wake/(T.getRadius())))))

    return Turbine_power


def print_wind_farm(turbine_string):
    """
    Inputs a list the represents the wind farm layout
    Prints the wind farm with an O where there is a turbine
    """
    wind_farm = getEmptyWindFarmList()

    T = Turbines()
    for i in range(400):
        if turbine_string[i] == True:
            T.addTurbine(math.floor(i/20), i % 20)

    turbine_list = T.getTurbines()
    for i in turbine_list:
        coords = i
        wind_farm[coords[0]][coords[1]] = i

    f = open('results.txt', 'a')

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
