import numpy as np
import carla


def get_weather_config(rng: np.random.RandomState) -> carla.WeatherParameters:
    result = carla.WeatherParameters()

    dry = rng.rand() < 0.5
    result.cloudiness = rng.uniform(low=0, high=100)
    result.precipitation = rng.uniform(low=0, high=100)
    result.precipitation_deposits = max(
        rng.uniform(low=0, high=100), result.precipitation - 10
    )  # puddles
    result.wind_intensity = rng.uniform(low=0, high=100)
    result.sun_azimuth_angle = rng.uniform(low=0, high=360)
    result.sun_altitude_angle = rng.uniform(low=-20, high=50)
    result.fog_density = rng.uniform(low=0, high=0)  # disable fog for now
    result.wetness = max(rng.uniform(low=0, high=100), result.precipitation - 10)

    if dry:
        result.precipitation = 0
        result.precipitation_deposits = 0
        result.wetness = 0

    return result
