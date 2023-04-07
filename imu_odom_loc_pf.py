# App: Hyper IMU

import numpy as np
import socket
import random
import matplotlib.pyplot as plt
import csv

class Particle:
    def __init__(self, x, y, theta, weight):
        self.x = x
        self.y = y
        self.theta = theta
        self.weight = weight
        self.gyro_bias = 0.0

class PFLocalization:
    def __init__(self):
        self.prev_time = 0.0

        self.lin_acc = [0.0, 0.0, 0.0]
        self.ang_vel = [0.0, 0.0, 0.0]

        self.num_particles = 2000
        self.particles = []
        self.weights = np.zeros(self.num_particles)
        self.weights.fill(1.0/self.num_particles)
        self.initialize_particles()
        self.last_gyro_theta = None
        self.alpha = 0.5 # Complementary filter constant

        self.dt = 0.01

        self.calib_cnt = 0
        self.calib_lin = np.array([0.0, 0.0, 0.0])
        self.calib_ang = np.array([0.0, 0.0, 0.0])

        # open the file in the write mode
        self.f = open('pose.csv', 'w', newline='')
        self.writer = csv.writer(self.f)

        # self.fig, self.ax = plt.subplots()
        # plt.ion()
        # plt.show()
    
    def initialize_particles(self):
        for i in range(self.num_particles):
            particle = Particle(np.random.uniform(-0.5,0.5),
                                np.random.uniform(-0.5,0.5),
                                np.random.uniform(-np.pi/4, np.pi/4),
                                1.0/self.num_particles)
            self.particles.append(particle)

    def update(self):
        z_acc = self.lin_acc
        z_gyro = self.ang_vel[2]
        dt = self.dt
        for i in range(self.num_particles):
            # Sample a new particle from the previous particles
            if self.last_gyro_theta is None:
                # self.last_gyro_theta = np.arctan2(-z_acc[0], np.sqrt(z_acc[1]**2 + z_acc[2]**2)) - self.particles[i].theta
                self.last_gyro_theta = z_gyro - self.particles[i].theta - self.particles[i].gyro_bias
            
            # Update the particle's position and angle based on the gyroscope reading
            self.particles[i].x += z_acc[0] * np.cos(self.particles[i].theta) * dt - z_acc[1] * np.sin(self.particles[i].theta) * dt ** 2 / 2
            self.particles[i].y += z_acc[0] * np.sin(self.particles[i].theta) * dt + z_acc[1] * np.cos(self.particles[i].theta) * dt ** 2 / 2

            # Compute the change in theta based on the gyroscope reading
            dtheta = z_gyro * dt - self.particles[i].gyro_bias # + np.random.normal(0, 0.01)
            self.particles[i].theta += dtheta

            # Apply the complementary filter to combine the accelerometer and gyroscope readings
            self.particles[i].theta = self.alpha * (self.particles[i].theta + np.arctan2(-z_acc[0], np.sqrt(z_acc[1]**2 + z_acc[2]**2))) + (1 - self.alpha) * (self.last_gyro_theta)
            self.last_gyro_theta = self.particles[i].theta
                    
            # Wrap the angle to [-pi, pi]
            self.particles[i].theta = np.arctan2(np.sin(self.particles[i].theta), np.cos(self.particles[i].theta))

            #  Update the particle's gyro bias
            self.particles[i].gyro_bias += np.random.normal(0, 0.01)
                    
            # Calculate the weight of the particle based on the difference between the predicted and actual measurements
            z_hat = np.array([np.sqrt(self.particles[i].x**2 + self.particles[i].y**2), # + np.random.normal(0, 0.01),
                            self.particles[i].theta + np.random.normal(0, 0.05)])
            diff = np.array([np.sqrt((z_hat[0] - np.sqrt(z_acc[0]**2 + z_acc[1]**2))**2 + (z_hat[1] - z_gyro)**2),
                            np.arctan2(np.sin(z_gyro - self.particles[i].theta - self.particles[i].gyro_bias),
                                        np.cos(z_gyro - self.particles[i].theta - self.particles[i].gyro_bias))])
            self.weights[i] *= np.exp(-0.5 * np.dot(diff, diff))
                    
        # Normalize the weights
        if np.sum(self.weights) > 0:
            self.weights /= np.sum(self.weights)
        else:
            self.weights[:] = 1.0 / self.num_particles

        # resample
        # Compute effective number of particles
        neff = 1 / np.sum(np.square(self.weights))

        # Resample only if neff is below a threshold
        if neff < self.num_particles / 5:
            self.resample()

        # plot
        # self.plot_particles()
    
    def plot_particles(self):
        # Plot particles
        xs = [particle.x for particle in self.particles]
        ys = [particle.y for particle in self.particles]
        self.ax.clear()
        self.ax.scatter(xs, ys, s=5)
        plt.draw()
        plt.pause(0.001)

    def get_estimated_pose(self):
        x_mean = 0.0
        y_mean = 0.0
        theta_mean = 0.0
        for particle in self.particles:
            x_mean += particle.x
            y_mean += particle.y
            theta_mean += particle.theta
        x_mean /= self.num_particles
        y_mean /= self.num_particles
        theta_mean /= self.num_particles
        return (x_mean, y_mean, theta_mean)
    
    def resample(self):            
        # Resample particles based on the weights
        new_particles = []
        for i in range(self.num_particles):
            index = np.random.choice(self.num_particles, p=self.weights)
            new_particle = Particle(self.particles[index].x,
                                        self.particles[index].y,
                                        self.particles[index].theta,
                                        self.weights[index])
            new_particle.gyro_bias = self.particles[index].gyro_bias
            new_particles.append(new_particle)
            
        self.particles = new_particles
        self.weights = np.array([1.0/self.num_particles for i in range(self.num_particles)])
    
    def calibrate(self):
        self.calib_lin = self.calib_lin + self.lin_acc
        self.calib_ang = self.calib_ang + self.ang_vel
    
    def run(self, sensor_data):
        dt = float(sensor_data[0]) - self.prev_time
        if dt >= 80 and dt <= 400:
            self.dt = dt/1000
            # self.lin_acc = np.array([float(sensor_data[1]), float(sensor_data[2]), 0.0])
            # self.ang_vel = np.array([0.0, 0.0, float(sensor_data[6])])
            self.lin_acc = np.array([round(float(sensor_data[1]), 3), round(float(sensor_data[2]), 3), 0.0]) # round(acc_z, 2)]
            self.ang_vel = np.array([0.0, 0.0, round(float(sensor_data[6]), 2)]) # [gyro_x, gyro_y, gyro_z]

            if self.calib_cnt <= 100:
                self.calibrate()
                if self.calib_cnt == 100:
                    self.calib_lin = self.calib_lin / (self.calib_cnt+1)
                    self.calib_ang = self.calib_ang / (self.calib_cnt+1)
                    print("calib done")
                self.calib_cnt += 1
                return

            self.lin_acc = self.lin_acc - self.calib_lin
            self.ang_vel = self.ang_vel - self.calib_ang
            
            self.update()
            if self.writer:
                self.writer.writerow(self.get_estimated_pose())
            # print(self.get_estimated_pose())

        self.prev_time = float(sensor_data[0])

if __name__ == "__main__":
    pfLoc = PFLocalization()
    HOST = '192.168.246.233'
    PORT = 2055
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        print(f"Listening for connections on {HOST}:{PORT}...")
        conn, addr = s.accept()
        with conn:
            print('Connected by', addr)
            while True:
                try:
                    data = conn.recv(1024)
                    if not data:
                        continue
                    sensor_data = data.decode().split(',')
                    if len(sensor_data) != 7:
                        continue
                    pfLoc.run(sensor_data)
                    conn.sendall(f"ACK".encode())
                except KeyboardInterrupt:
                    print("Program Killed. Exiting!!")
                    break
                except ConnectionResetError:
                    print("connection lost")
                    break
                except ValueError:
                    pass


