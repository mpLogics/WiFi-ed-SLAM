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

        self.alpha = 1.0    # complementary filter coefficient

        self.calib_cnt = 0
        self.calib_lin = np.array([0.0, 0.0, 0.0])
        self.calib_ang = np.array([0.0, 0.0, 0.0])

        # Define constants
        self.N_Particles = 1
        self.dt = 0.01
        self.sigma_velocity = 0.05
        self.sigma_position = 0.1

        # Define state vector
        self.state = np.zeros((3, 1))
        # self.state[:3] = np.random.uniform(-1, 1, size=(3, 1))

        # Initialize particles with Gaussian distribution around the initial position
        self.mu = np.array([[0], [0], [0]])
        self.cov = np.diag([1, 1, 1])
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

        x, y, theta = float(self.state[0]), float(self.state[1]), float(self.state[2])

        theta = self.alpha * (theta + gyro_rate[2] * self.dt) + (1 - self.alpha) * math.atan2(math.sin(theta), math.cos(theta))
        theta = self.wrapTopi(theta)

        # calculate displacement
        dx = (acceleration[0] * math.cos(theta) + acceleration[1] * math.sin(theta)) * self.dt
        dy = (-acceleration[0] * math.sin(theta) + acceleration[1] * math.cos(theta)) * self.dt
        # print(acceleration)
        # print(gyro_rate)
        # print(f"{dx}, {dy}, {theta}")

        # update position
        x += dx
        y += dy

        return np.array([x, y, theta])


    # Define measurement model function
    def measurement_model(self, state):
        measurement_noise = np.random.normal(scale=self.sigma_position, size=(2, 1))
        return state[0:2] + measurement_noise
    
    # Define update function
    def update(self):
        for i in range(self.N_Particles):
            self.particles[i, :] = self.motion_model().flatten()
        # position_estimate = np.mean(self.particles, axis=0)
        # print(f"p:{position_estimate[0], position_estimate[1], position_estimate[2]}")
    
        for i in range(self.N_Particles):
            state = self.particles[i, :]
            predicted_measurement = self.measurement_model(state)
            distance = np.linalg.norm(predicted_measurement - self.state[0:2])
            self.weights[i] *= np.exp(-0.5 * (distance ** 2) / (self.sigma_position ** 2))
        self.weights /= np.sum(self.weights)

        # if np.sum(self.weights) > 0:
        #     self.weights /= np.sum(self.weights)
        # else:
        #     self.weights[:] = 1.0 / self.N_Particles

        # Compute effective number of particles
        neff = 1 / np.sum(np.square(self.weights))

        # Resample only if neff is below a threshold
        if neff < self.N_Particles / 5:
            self.resample()

    # Define resampling function
    def resample(self):
        print("resampling")
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
        self.particles, self.weights = new_particles, new_weights

    def calibrate(self):
        self.calib_lin = self.calib_lin + self.linear_acceleration
        self.calib_ang = self.calib_ang + self.angular_velocity

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
                            self.linear_acceleration = np.array([round(float(sensor_data[4]), 3), round(float(sensor_data[5]), 3), 0.0]) # round(acc_z, 2)]
                            self.angular_velocity = np.array([0.0, 0.0, round(float(sensor_data[3]), 6)]) # [gyro_x, gyro_y, gyro_z]
                            print(f"{sensor_data[0]}, {sensor_data[1]}, {sensor_data[2]}, {sensor_data[3]}, {sensor_data[4]}, {sensor_data[5]}, {sensor_data[6]}")

                            if self.calib_cnt <= 50:
                                self.calibrate()
                                if self.calib_cnt == 50:
                                    # print(self.calib_lin)
                                    # print(self.calib_ang)
                                    self.calib_lin = self.calib_lin / (self.calib_cnt+1)
                                    self.calib_ang = self.calib_ang / (self.calib_cnt+1)
                                    # print(self.calib_lin)
                                    # print(self.calib_ang)
                                    # print("calib done")
                                self.calib_cnt += 1
                                continue

                            self.linear_acceleration = self.linear_acceleration - self.calib_lin
                            self.angular_velocity = self.angular_velocity - self.calib_ang

                            # Update particles and weights
                            self.update()

                            # Output current estimate of position
                            self.position_estimate = np.mean(self.particles, axis=0)
                            print(f"t:{dt}, lin:{self.linear_acceleration}, ang:{self.angular_velocity}, p:{self.position_estimate}")
                            # print(f"p:{self.position_estimate[0], self.position_estimate[1], self.position_estimate[2]}")

                            # Plot particles
                            # ax.clear()
                            # ax.set_xlim([-20, 20])
                            # ax.set_ylim([-20, 20])
                            # # ax.set_zlim([-2, 2])
                            # ax.scatter(self.particles[:, 0], self.particles[:, 1], s=1, c='r', alpha=0.1)
                            # ax.scatter(self.state[0], self.state[1], s=100, c='b', alpha=1.0)

                            # plt.draw()
                            # plt.pause(0.01)

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
                    except ValueError:
                        pass
if __name__ == "__main__":
    pfLoc = PFLocalization()
    pfLoc.run()
