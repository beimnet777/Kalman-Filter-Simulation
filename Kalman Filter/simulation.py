import numpy as np

class Simulation2D:
    def __init__(self, initial_position, initial_velocity, process_noise_std,measurement_noise_std ):
        self.measurement_noise_std = measurement_noise_std
        self.position = np.array(initial_position, dtype = np.float64)  # [lat, long]
        self.velocity = np.array(initial_velocity, dtype = np.float64)  # [velocity_lat, velocity_long]
        self.process_noise_std = process_noise_std
        self.history = {'position': []}

    def update(self):
        self.position += self.velocity
        self.position += np.random.normal(0, self.process_noise_std, 2)
        self.history['position'].append(self.position.copy())


    def get_measurement(self):
        return self.position + np.random.normal(0, self.measurement_noise_std, 2)
    

