import wake_model

Turbines = [1]*100


wind_speed = 1
wind_direction = 180


for i in range(45,405,90):
    PowerOutput = wake_model.Wake(
        Turbines,
        wind_speed,
        i,
        10, 10)
    print(i)

    print(PowerOutput)
    wake_model.print_wind_farm(Turbines, 10, 10)
