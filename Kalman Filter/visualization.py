import pygame
import numpy as np
import matplotlib.pyplot as plt

class VehicleSimulation:
    def __init__(self, initial_position, initial_velocity, process_noise_std):
        self.position = initial_position
        self.velocity = initial_velocity
        self.process_noise_std = process_noise_std
        self.history = {'position': [], 'velocity': []}

    def update(self):
        self.position += self.velocity
        self.position += np.random.normal(0, self.process_noise_std)

        self.history['position'].append(self.position)
        self.history['velocity'].append(self.velocity)
        return self.position


class KalmanFilter:
    def __init__(self, initial_state, initial_covariance, process_noise_std, measurement_noise_std):
        self.state = np.array(initial_state)
        self.covariance = np.array(initial_covariance)
        self.process_noise_std = process_noise_std
        self.measurement_noise_std = measurement_noise_std
        self.history = {'position': [], 'velocity': []}

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
        self.history['position'].append(self.state[0])
        self.history['velocity'].append(self.state[1])




# Pygame visualization setup
class KalmanFilterVisualization:
    def __init__(self, screen_size, vehicle_sim, kalman_filter):
        pygame.init()
        self.screen = pygame.display.set_mode(screen_size)
        self.vehicle_sim = vehicle_sim
        self.kalman_filter = kalman_filter
        self.clock = pygame.time.Clock()
        self.running = True
    

    def update_plots(self):
        fig, (ax1, ax2) = plt.subplots(2, figsize=(5, 4))
        
        ax1.plot(self.vehicle_sim.history['position'], label='Actual Position')
        ax1.plot(self.kalman_filter.history['position'], label='Estimated Position')
        ax1.legend()

        ax2.plot(self.vehicle_sim.history['velocity'], label='Actual Velocity')
        ax2.plot(self.kalman_filter.history['velocity'], label='Estimated Velocity')
        ax2.legend()

        # Save the plot to a file
        plt.savefig('plot.png')
        plt.close(fig)

    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            # Update vehicle simulation and Kalman filter
            measurement = self.vehicle_sim.update()
            self.kalman_filter.predict()
            self.kalman_filter.update(measurement)


            self.update_plots()

            self.screen.fill((255, 255, 255))

            plot_image = pygame.image.load('plot.png')
            plot_image = pygame.transform.scale(plot_image, (600, 500))
            self.screen.blit(plot_image, (400, 0))

            # Clear screen
            

            # Draw vehicle position
            pygame.draw.circle(self.screen, (0, 255, 0), (int(self.vehicle_sim.position * 10), 250), 10)

            # Draw estimated position by Kalman filter
            pygame.draw.circle(self.screen, (255, 0, 0), (int(self.kalman_filter.state[0] * 10), 250), 10)

            # Update display
            pygame.display.flip()
            self.clock.tick(3)

        pygame.quit()

# Create and run the simulation
screen_size = (1000, 800)
vehicle_sim = VehicleSimulation(initial_position=0, initial_velocity=1, process_noise_std=0.1)
kf = KalmanFilter(initial_state=[0, 1], initial_covariance=[[1, 0], [0, 1]], process_noise_std=0.1, measurement_noise_std=1)
visualization = KalmanFilterVisualization(screen_size, vehicle_sim, kf)
visualization.run()
