from .car import VehicleSimulation
class KalmanFilter:
    def __init__(self, initial_state, initial_covariance, process_noise_std, measurement_noise_std):
        self.state = np.array(initial_state)
        self.covariance = np.array(initial_covariance)
        self.process_noise_std = process_noise_std
        self.measurement_noise_std = measurement_noise_std

    def predict(self):
        # Predict the next state
        self.state[0] += self.state[1]
        self.covariance[0, 0] += self.process_noise_std ** 2

    def update(self, measurement):
        # Kalman gain
        K = self.covariance[0, 0] / (self.covariance[0, 0] + self.measurement_noise_std ** 2)
        
        # Update state and covariance
        self.state[0] += K * (measurement - self.state[0])
        self.covariance[0, 0] *= (1 - K)


vehicle_sim = VehicleSimulation(initial_position=0, initial_velocity=1, process_noise_std=0.1)


kf = KalmanFilter(initial_state=[0, 1], initial_covariance=[[1, 0], [0, 1]], process_noise_std=0.1, measurement_noise_std=1)
measurement = vehicle_sim.update()
kf.predict()
kf.update(measurement)
print("Estimated position:", kf.state[0])
