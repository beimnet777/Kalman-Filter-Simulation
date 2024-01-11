import pygame
import numpy as np
import matplotlib.pyplot as plt
from Kalman import KalmanFilter2D
from simulation import Simulation2D



class KalmanFilterVisualization:
    def __init__(self, screen_size, vehicle_sim, kalman_filter):
        pygame.init()
        self.clock = pygame.time.Clock()
        self.running = True

        # Colors and other constants
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)
        self.SCALE = 10  # Scale factor for displaying positions
        self.screen = pygame.display.set_mode(screen_size)
        self.vehicle_sim = vehicle_sim
        self.kalman_filter = kalman_filter
        self.clock = pygame.time.Clock()
    

    def update_plots(self):
        fig, (ax1, ax2) = plt.subplots(2, figsize=(5, 4))

        actual_x_position = [ pos[0] for pos in self.vehicle_sim.history['position']]
        actual_y_position = [ pos[1] for pos in self.vehicle_sim.history['position']]
        print(actual_x_position)
        estimated_x_position = [ pos[0] for pos in self.kalman_filter.history['position']]
        estimated_y_position = [ pos[1] for pos in self.kalman_filter.history['position']]
        
        ax1.plot(actual_x_position, label='Actual X Position')
        ax1.plot(estimated_x_position, label='Estimated X Position')
        ax1.legend()

        ax2.plot(actual_y_position, label='Actual Y Position')
        ax2.plot(estimated_y_position, label='Estimated Y Postion')
        ax2.legend()

        # Save the plot to a file
        plt.savefig('plot.png')
        plt.close(fig)
    
    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RIGHT:

            # Update the simulation and Kalman filter
            
                        self.vehicle_sim.update()
                        measurement = self.vehicle_sim.get_measurement()
                        self.kalman_filter.predict()
                        self.kalman_filter.update(measurement)

                        self.update_plots()


            # Clear the screen
            self.screen.fill(self.WHITE)

            plot_image = pygame.image.load('plot.png')
            plot_image = pygame.transform.scale(plot_image, (600, 500))
            self.screen.blit(plot_image, (1000, 0))


            # Draw the actual position from the simulation
            actual_pos = (int(self.vehicle_sim.position[0] * self.SCALE), int(self.vehicle_sim.position[1] * self.SCALE))
            pygame.draw.circle(self.screen, self.RED, actual_pos, 10)

            # Draw the estimated position from the Kalman filter
            estimated_pos = (int(self.kalman_filter.state[0] * self.SCALE), int(self.kalman_filter.state[2] * self.SCALE))
            pygame.draw.circle(self.screen, self.BLUE, estimated_pos, 10)

            # Update the display
            pygame.display.flip()
            self.clock.tick(1)

        pygame.quit()


# Simulation and Kalman Filter setup
simulation = Simulation2D([15, 15], [3,2], 0.1, 0.05)

initial_state = [0, 0, 0, 0]  # [lat_position, lat_velocity, long_position, long_velocity]
initial_covariance = np.eye(4) * 0.1  
process_noise_covariance = np.eye(4) * 0.1  
measurement_noise_covariance = np.eye(2) * 0.5 

kalman_filter = KalmanFilter2D(initial_state, initial_covariance, process_noise_covariance, measurement_noise_covariance)

Kalman_Visualization = KalmanFilterVisualization((1600,800) , simulation, kalman_filter )

Kalman_Visualization.run()