import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import socket
import csv
import sys

class Particle:
    def __init__(self, x_init, x_std, y_init, y_std, theta_init, theta_std, N):
        self.particles_x = np.random.normal(x_init,x_std,N)
        self.particles_y = np.random.normal(y_init,y_std,N)
        self.particles_theta = np.random.normal(theta_init,theta_std,N)

    def get_particles(self):
        return np.vstack((self.particles_x, self.particles_y, self.particles_theta)).T

class PFLocalization:
    def __init__(self):
        self.N = 1000 #num of particles

        #prior
        self.x_init = 0
        self.y_init = 0
        self.theta_init = 0

        #standard deviation
        self.x_std = 0.5
        self.y_std = 0.5
        self.theta_std = np.pi/100

        self.p_obj = Particle(self.x_init, self.x_std, self.y_init, self.y_std, self.theta_init, self.theta_std, self.N)
        self.particles = self.p_obj.get_particles()

        self.weights = np.ones(self.N) / self.N

        self.prev_time = 0.0

        self.lin_acc = np.zeros(3)
        self.ang_vel = np.zeros(3)
        self.dt = 0.0

        self.calib_cnt = 0
        self.calib_lin = np.array([0.0, 0.0, 0.0])
        self.calib_ang = np.array([0.0, 0.0, 0.0])

        # open the file in the write mode
        self.f = open('pose_imu_pf.csv', 'w', newline='')
        self.writer = csv.writer(self.f)

        self.real_time = False

    def low_variance_resample(self):
        indexes = np.zeros(self.N, 'i')
        r = np.random.uniform(0, 1/self.N)
        c = self.weights[0]
        i = 0
        for m in range(self.N):
            u = r + m/self.N
            while u > c:
                i += 1
                c += self.weights[i]
            indexes[m] = i
        return indexes
    
    def systematic_resample(self):
        indexes = np.zeros(self.N, 'i')
        cumulative_sum = np.cumsum(self.weights)
        interval_size = cumulative_sum[-1] / self.N
        pointer = np.random.uniform(0, interval_size)
        j = 0
        for m in range(self.N):
            while pointer > cumulative_sum[j]:
                j += 1
            indexes[m] = j
            pointer += interval_size
        return indexes

    def resample(self):
        indexes = self.low_variance_resample()
        particles_resampled = self.particles[indexes]
        weights_resampled = self.weights[indexes]
        self.particles = particles_resampled
        self.weights = weights_resampled
    
    def predict(self, dt, acc_data, angular_velocity):
        linear_velocity = acc_data * dt
        delta_theta = angular_velocity[2] * dt
        # update x and y positions
        self.particles[:, 0] += ((linear_velocity[0] * np.cos(self.particles[:, 2])) - (linear_velocity[1] * np.sin(self.particles[:, 2]))) * dt
        self.particles[:, 1] += ((linear_velocity[0] * np.sin(self.particles[:, 2])) + (linear_velocity[1] * np.cos(self.particles[:, 2]))) * dt
        # update heading
        self.particles[:, 2] += delta_theta
        self.particles[:, 2] = (self.particles[:, 2] + np.pi) % (2 * np.pi) - np.pi
    
    def update(self): # , z):
        pass
    
    def get_pose(self):
        x = np.sum(self.particles[:, 0] * self.weights)
        y = np.sum(self.particles[:, 1] * self.weights)
        theta = np.sum(self.particles[:, 2] * self.weights)
        return x, y, theta
    
    def calibrate(self):
        self.calib_lin = self.calib_lin + self.lin_acc
        self.calib_ang = self.calib_ang + self.ang_vel

    def run(self, sensor_data):
        dt = (float(sensor_data[0]) - self.prev_time)/1000
        if dt >= 0.080 and dt <= 0.400:
            self.dt = dt
            self.lin_acc = np.array([round(float(sensor_data[1]), 2), round(float(sensor_data[2]), 2), 0.0]) # round(acc_z, 2)]
            self.ang_vel = np.array([0.0, 0.0, round(float(sensor_data[6]), 1)]) # [gyro_x, gyro_y, gyro_z]

            if self.calib_cnt <= 100 and self.real_time:
                self.calibrate()
                if self.calib_cnt == 100:
                    self.calib_lin = self.calib_lin / (self.calib_cnt+1)
                    self.calib_ang = self.calib_ang / (self.calib_cnt+1)
                    print("calib done")
                self.calib_cnt += 1
                return

            self.lin_acc = self.lin_acc - self.calib_lin
            # self.ang_vel = self.ang_vel - self.calib_ang

            # Apply linear acceleration measurements
            self.predict(self.dt, self.lin_acc, self.ang_vel)

            self.update()

            # Resampling
            # self.resample()

            # Publish the pose
            pose = self.get_pose()
            print(pose)
            self.writer.writerow([float(sensor_data[0]), pose[0], pose[1], pose[2]])
        self.prev_time = float(sensor_data[0])

    def end(self):
        self.f.close()
    

if __name__ == "__main__":
    pfLoc = PFLocalization()
    if len(sys.argv) > 1:
        pfLoc.real_time = False
        with open(sys.argv[1], mode ='r', newline='') as file:
            csvFile = csv.reader(file)
            time = 0.0
            for lines in csvFile:
                sensor_data = np.array([time,
                                        float(lines[1]), float(lines[2]), float(lines[3]),
                                        float(lines[4]), float(lines[5]), float(lines[6])])
                time = time + 100
                pfLoc.run(sensor_data)
                
    else:
        pfLoc.real_time = True
        HOST = '192.168.112.233'
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