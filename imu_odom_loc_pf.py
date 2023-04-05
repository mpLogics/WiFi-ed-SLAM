# App: Hyper IMU

import numpy as np
import random
import socket
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class PFLocalization:
    def __init__(self):
        self.HOST = '192.168.246.233'
        self.PORT = 2055

        self.prev_time = 0.0

        self.linear_acceleration = [0.0, 0.0, 0.0]
        self.angular_velocity = [0.0, 0.0, 0.0]

        # Define constants
        self.N_Particles = 1000
        self.dt = 0.01
        self.sigma_velocity = 0.05
        self.sigma_position = 0.1

        # Define state vector
        self.state = np.zeros((6, 1))
        # self.state[:3] = np.random.uniform(-1, 1, size=(3, 1))

        # Initialize particles with Gaussian distribution around the initial position
        self.mu = np.array([[0], [0], [0], [0], [0], [0]])
        self.cov = np.diag([1, 1, 1, 1, 1, 1])
        self.particles = np.random.multivariate_normal(self.mu.ravel(), self.cov, size=self.N_Particles)

        # Initialize weights to be uniform
        self.weights = np.ones(self.N_Particles) / self.N_Particles
    
    def wrapTopi(self, angle):
        phase = (angle + np.pi) % (2*np.pi) - np.pi
        return phase

    def motion_model(self):
        # velocity_noise = np.random.normal(scale=self.sigma_velocity, size=(3, 1))
        acceleration = np.array([[self.linear_acceleration[0]], [self.linear_acceleration[1]], [self.linear_acceleration[2]]])
        gyro_rate = np.array([[self.angular_velocity[0]], [self.angular_velocity[1]], [self.angular_velocity[2]]])

        # Compute rotation matrix from roll, pitch, and yaw angles
        roll, pitch, yaw = float(self.state[3]), float(self.state[4]), float(self.state[5])
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(roll), -np.sin(roll)],
                       [0, np.sin(roll), np.cos(roll)]])
        Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                       [0, 1, 0],
                       [-np.sin(pitch), 0, np.cos(pitch)]])
        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                       [np.sin(yaw), np.cos(yaw), 0],
                       [0, 0, 1]])
        R = np.dot(Rz, np.dot(Ry, Rx))

        # compute acceleration in the global frame
        acceleration_global = np.dot(R, acceleration)

        # add noise to velocity and position
        velocity_noise = np.random.normal(scale=self.sigma_velocity, size=(3, 1))
        position_noise = np.random.normal(scale=self.sigma_position, size=(3, 1))

        # compute velocity
        v_x, v_y, v_z = self.state[3:6]
        velocity = np.array([v_x, v_y, v_z]) + acceleration_global * self.dt + velocity_noise

        # compute position
        position = np.array([self.state[0], self.state[1], self.state[2]]) + velocity * self.dt + position_noise
        # position[0] = round(float(position[0]), 4)
        # position[1] = round(float(position[1]), 4)
        position[2] = 0.0 # round(float(position[2]), 4)

        # compute angular rates
        p, q, r = gyro_rate - np.array([pitch, roll, yaw]).reshape(3,1)

        # compute new state
        new_state = np.array([position[0], position[1], position[2], self.wrapTopi(roll + p*self.dt), self.wrapTopi(pitch + q*self.dt), self.wrapTopi(yaw + r*self.dt)])
        return new_state


    # Define measurement model function
    def measurement_model(self, state):
        measurement_noise = np.random.normal(scale=self.sigma_position, size=(3, 1))
        return state[0:3] + measurement_noise
    
    # Define update function
    def update(self):
        for i in range(self.N_Particles):
            state = self.particles[i, :]
            predicted_measurement = self.measurement_model(state)
            distance = np.linalg.norm(predicted_measurement - self.state[0:3])
            self.weights[i] *= np.exp(-0.5 * (distance ** 2) / (self.sigma_position ** 2))

            self.particles[i, :] = self.motion_model().flatten()
        if np.sum(self.weights) > 0:
            self.weights /= np.sum(self.weights)
        else:
            self.weights[:] = 1.0 / self.N_Particles

    # Define resampling function
    def resample(self):
        new_particles = np.zeros_like(self.particles)
        new_weights = np.zeros_like(self.weights)
        cumulative_sum = np.cumsum(self.weights)
        r = random.uniform(0, 1 / self.N_Particles)
        j = 0
        for i in range(self.N_Particles):
            u = r + i / self.N_Particles
            while u > cumulative_sum[j]:
                j += 1
            new_particles[i, :] = self.particles[j, :]
            new_weights[i] = 1 / self.N_Particles
        # Compute effective number of particles
        neff = 1 / np.sum(np.square(self.weights))

        # Resample only if neff is below a threshold
        if neff < self.N_Particles / 2:
            self.particles, self.weights = new_particles, new_weights
        else:
            self.weights[:] = 1.0 / self.N_Particles
        
        return new_particles, new_weights

    def run(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((self.HOST, self.PORT))
            s.listen()
            print(f"Listening for connections on {self.HOST}:{self.PORT}...")
            conn, addr = s.accept()
            with conn:
                print('Connected by', addr)
                fig = plt.figure()
                ax = fig.add_subplot(111) # , projection='3d')
                ax.set_xlim([-20, 20])
                ax.set_ylim([-20, 20])
                # ax.set_zlim([-2, 2])
                while True:
                    try:
                        data = conn.recv(1024)
                        if not data:
                            continue
                        sensor_data = data.decode().split(',')
                        if len(sensor_data) != 7:
                            continue
                        # timestamp, gyro_x, gyro_y, gyro_z, acc_x, acc_y, acc_z = map(float, data.decode().split(','))
                        dt = float(sensor_data[0]) - self.prev_time
                        if dt >= 80 and dt <= 400:
                            self.dt = dt*(1/1000)
                            self.linear_acceleration = [round(float(sensor_data[4]), 3), round(float(sensor_data[5]), 3), 0.0] # round(acc_z, 2)]
                            self.angular_velocity = [0.0, 0.0, round(float(sensor_data[3]), 6)] # [gyro_x, gyro_y, gyro_z]

                            # Compute motion model
                            self.state = self.motion_model()
                            # print(f"s:{self.state[0], self.state[1], self.state[5]}")

                            # Update particles and weights
                            self.update()

                            # Resample particles
                            self.particles, self.weights = self.resample()

                            # Output current estimate of position
                            self.position_estimate = np.mean(self.particles[:, 0:6], axis=0)
                            # print(f"t:{dt}, lin:{self.linear_acceleration}, ang:{self.angular_velocity}, p:{self.position_estimate}")
                            # print(f"p:{self.position_estimate[0], self.position_estimate[1], self.position_estimate[5]}")

                            # Plot particles
                            ax.clear()
                            ax.set_xlim([-20, 20])
                            ax.set_ylim([-20, 20])
                            # ax.set_zlim([-2, 2])
                            ax.scatter(self.particles[:, 0], self.particles[:, 1], s=1, c='r', alpha=0.1)
                            ax.scatter(self.state[0], self.state[1], s=100, c='b', alpha=1.0)

                            plt.draw()
                            plt.pause(0.01)

                        self.prev_time = float(sensor_data[0])
                        conn.sendall(f"ACK".encode())
                    except KeyboardInterrupt:
                        plt.close()
                        print("Program Killed. Exiting!!")
                        break
                    except ConnectionResetError:
                        plt.close()
                        print("connection lost")
                        break
                    except Exception as e:
                        print(str(e))
                        plt.close()
                        continue
if __name__ == "__main__":
    pfLoc = PFLocalization()
    pfLoc.run()

