
import numpy as np
class KalmanFilter2D:
    def __init__(self, initial_state, initial_covariance, process_noise_covariance, measurement_noise_covariance):
        self.state = initial_state
        self.covariance = initial_covariance
        self.process_noise_covariance = process_noise_covariance
        self.measurement_noise_covariance = measurement_noise_covariance
        self.history = {'position': []}

    def predict(self):
        # State transition matrix for 2D
        dt = 1  # time step
        F = np.array([[1, dt, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, dt],
                      [0, 0, 0, 1]], dtype= np.float64
                      )

        # Predict next state
        self.state = F @ self.state
        self.covariance = F @ self.covariance @ F.T + self.process_noise_covariance

    def update(self, measurement):
        # Measurement matrix for 2D
        H = np.array([[1, 0, 0, 0],
                      [0, 0, 1, 0]] , dtype= np.float64)

        # Kalman Gain
        S = H @ self.covariance @ H.T + self.measurement_noise_covariance
        K = self.covariance @ H.T @ np.linalg.inv(S)

        # Update state
        y = measurement - H @ self.state  # Measurement residual
        self.state += K @ y
        self.covariance = (np.eye(4) - K @ H) @ self.covariance

        self.history['position'].append((self.state[0], self.state[2]))