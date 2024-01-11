import numpy as np

class VehicleSimulation:
    def __init__(self, initial_position, initial_velocity, process_noise_std):
        self.position = initial_position
        self.velocity = initial_velocity
        self.process_noise_std = process_noise_std

    def update(self):
        # Update vehicle position with some process noise
        self.position += self.velocity
        self.position += np.random.normal(0, self.process_noise_std)
        return self.position


