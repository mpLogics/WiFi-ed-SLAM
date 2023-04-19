import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import socket, math
import csv, random
import sys, time
from idw_interpolation import idw_interpolation, gmm_extr

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
        self.x_std = 0.1
        self.y_std = 0.1
        self.theta_std = np.pi/50

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
        self.f = open('data/pose_imu_pf.csv', 'w', newline='')
        self.writer = csv.writer(self.f)

        self.real_time = False

        self.gt_x = np.array([])
        self.gt_y = np.array([])
        self.gt_x_actual = np.array([])
        self.gt_y_actual = np.array([])
        self.leftX = 0.0
        self.rightX = 0.0
        self.leftY = 0.0
        self.rightY = 0.0
        self.total_time = 0.0
        self.current_iter = 0

        self.mean = np.zeros(3)
        self.predicted_mean = np.zeros(3)

    def resample(self, value):
        new_particles = np.zeros_like(self.particles)
        new_weights = np.zeros_like(self.weights)
        cumulative_sum = np.cumsum(self.weights)
        r = random.uniform(0, 1 / self.N)
        j = 0
        for i in range(self.N):
            u = r + i / self.N
            while u > cumulative_sum[j]:
                j += 1
            new_particles[i, :] = self.particles[j, :]
            new_particles[i, 0:2] += value
            new_weights[i] = 1 / self.N
        self.particles, self.weights = new_particles, new_weights
    
    def predict(self, dt, acc_data, angular_velocity):
        linear_velocity = acc_data * dt
        linear_velocity[0] = round(linear_velocity[0], 1)
        delta_theta = round(angular_velocity[2] * dt, 3)
        # print(f"{linear_velocity} - {angular_velocity} - {delta_theta}")
        # update x and y positions
        self.particles[:, 0] += ((linear_velocity[0] * np.cos(self.particles[:, 2])) - (linear_velocity[1] * np.sin(self.particles[:, 2]))) * dt
        self.particles[:, 1] += ((linear_velocity[0] * np.sin(self.particles[:, 2])) + (linear_velocity[1] * np.cos(self.particles[:, 2]))) * dt
        # update heading
        self.particles[:, 2] += np.round(delta_theta, 3)
        self.particles[:, 2] = (self.particles[:, 2] + np.pi) % (2 * np.pi) - np.pi

    def update(self):
        # Calculate the expected measurement mean and covariance based on the particles
        pred_mean, pred_cov = self.get_pose()

        # Compute the mean and covariance of the Wi-Fi signals
        wifi_centers, wifi_covs = gmm_extr(self.interp_xi, self.interp_yi, self.interp_zi, self.local_max_num, threshold=0.55)

        # Find the nearest Wi-Fi signal to the predicted mean of the particles
        current_min_dist = 1e7
        nearest_mode_mean = np.zeros(2)
        nearest_mode_cov = np.zeros([2, 2])
        for i in range(len(wifi_centers)):
            dist = np.linalg.norm([pred_mean[0] - wifi_centers[i][0], pred_mean[1] - wifi_centers[i][1]])
            if dist < current_min_dist:
                current_min_dist = dist
                nearest_mode_mean = np.array(wifi_centers[i]).reshape(1, 2)
                nearest_mode_cov = np.array(wifi_covs[i])

        # Update the weight of each particle based on its distance to the nearest Wi-Fi signal
        for i in range(len(self.particles)):
            # noise_std = 0.1
            # predicted_weightage = 0.5
            # wifi_weightage = 0.5

            distance = np.linalg.norm(self.particles[i, 0:2] - nearest_mode_mean)
            likelihood = np.exp(-0.5*distance**2 / 0.5)
            self.weights[i] *= likelihood
    
            wifi_contribution = np.array([0.2 * (nearest_mode_mean - self.particles[i, 0:2])]).reshape(2,)
            predicted_mean_contribution = np.array([0.8 * (pred_mean[0:2] - self.particles[i, 0:2])]).reshape(2,)
            # self.particles[i, 0:2] += wifi_contribution + predicted_mean_contribution

        # Resample the particles based on the updated weights
        # Normalize the weights
        if np.sum(self.weights) > 0:
            self.weights /= np.sum(self.weights)
        else:
            self.weights[:] = 1.0 / self.N

        # # resample
        # # Compute effective number of particles
        neff = 1 / np.sum(np.square(self.weights))

        # Resample only if neff is below a threshold
        if neff < self.N / 5:
            self.resample(wifi_contribution + predicted_mean_contribution)

    
    def get_pose(self):
        mean = np.zeros(3)
        mean[0] = np.sum(self.particles[:, 0] * self.weights)
        mean[1] = np.sum(self.particles[:, 1] * self.weights)
        mean[2] = np.sum(self.particles[:, 2] * self.weights)
        particles = np.array([self.particles[:, 0], self.particles[:, 1], self.particles[:, 2]])
        cov = np.cov(particles)
        return mean, cov
    
    def calibrate(self):
        self.calib_lin = self.calib_lin + self.lin_acc
        self.calib_ang = self.calib_ang + self.ang_vel

    def simulate_wifi(self):
        global_max = [self.gt_x[self.current_iter], self.gt_y[self.current_iter]]
        # print(f"global: {self.gt_x_actual[self.current_iter]}, {self.gt_y_actual[self.current_iter]}")
        local_max_num = random.randint(1, 2)
        existing_points_num = random.randint(20, 30)

        spead_dorder = 0.5        #local maxima may outside of map border
        dist = 0.2        #local maxima should at least away from global one this distance

        existing_points = []    # x, y, possibility

        global_max.append(random.uniform(0.85, 0.91))       #MAX_POSSIBILITY
        existing_points.append(global_max)
        
        for j in range(local_max_num + existing_points_num):
            # create points position
            temp_dist = 0
            while temp_dist < dist:
                xi = random.uniform(self.leftX - spead_dorder, self.rightX + spead_dorder)
                yi = random.uniform(self.leftY - spead_dorder, self.rightY + spead_dorder)
                temp_dist = np.linalg.norm([xi - global_max[0], yi - global_max[1]])

            # assign possibility for each existing points
            if j < local_max_num:   # local maxima
                existing_points.append([xi, yi, random.uniform(0.5, 0.69)])
            elif j < (local_max_num + int(existing_points_num / 4)):    # small group of some possibilities
                existing_points.append([xi, yi, random.uniform(0.20, 0.35)])
            else:   #other possibilities
                existing_points.append([xi, yi, random.uniform(0.04, 0.17)])

        self.interp_xi = np.linspace(self.leftX - spead_dorder, self.rightX + spead_dorder, 50)
        self.interp_yi = np.linspace(self.leftY - spead_dorder, self.rightY + spead_dorder, 50)
        zi1, zi2, zi3 = list(zip(*([(q[0], q[1], q[2]) for q in existing_points])))

        self.interp_zi = idw_interpolation(zi1, zi2, self.interp_xi, self.interp_yi, zi3, 2)

        rows = len(self.interp_zi)
        cols = len(self.interp_zi[0])
        self.interp_zi = np.array(self.interp_zi).reshape(rows, cols)
        self.local_max_num = local_max_num
    
    def plot(self):
        plt.figure()
        plt.contourf(self.interp_xi, self.interp_yi, self.interp_zi, levels=15, cmap='GnBu')
        plt.plot(self.gt_x_actual, self.gt_y_actual)
        plt.plot(self.gt_x_actual, self.gt_y_actual,'x')
        plt.plot(self.mean[:, 0], self.mean[:, 1])
        plt.scatter(self.particles[:, 0], self.particles[:, 1], s=0.5, color='red')
        plt.colorbar(label='Possibility')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title(f'Possibility Distribution')
        # plt.legend()
        plt.show()
    
    def final_plot(self):
        plt.figure()
        print(self.mean.shape)
        plt.plot(self.mean[:, 0], self.mean[:, 1])
        plt.plot(self.predicted_mean[:, 0], self.predicted_mean[:, 1])
        plt.plot(self.gt_x_actual, self.gt_y_actual)
        plt.plot(self.gt_x_actual, self.gt_y_actual,'x')
        plt.show()

    def run(self, sensor_data):
        dt = (float(sensor_data[0]) - self.prev_time)/1000
        self.total_time = self.total_time + dt
        if dt >= 0.080 and dt <= 0.400:
            self.dt = dt
            self.lin_acc = np.array([float(sensor_data[1])*10, 0.0, 0.0]) # round(acc_z, 2)]
            self.ang_vel = np.array([0.0, 0.0, float(sensor_data[6])]) # [gyro_x, gyro_y, gyro_z]

            if self.calib_cnt <= 100 and self.real_time:
                self.calibrate()
                if self.calib_cnt == 100:
                    self.calib_lin = self.calib_lin / (self.calib_cnt+1)
                    self.calib_ang = self.calib_ang / (self.calib_cnt+1)
                    print("calib done")
                    self.total_time = 0.0
                self.calib_cnt += 1
                return

            self.lin_acc = self.lin_acc - self.calib_lin
            # self.ang_vel = self.ang_vel - self.calib_ang

            if (self.lin_acc[0] < 0.0):
                self.lin_acc[0] = 0.0

            # Apply linear acceleration measurements
            self.predict(self.dt, self.lin_acc, self.ang_vel)
            # if self.total_time >= 0.9:
            self.simulate_wifi()
            self.current_iter = self.current_iter + 1
            self.update()
            # print(self.total_time)
            self.total_time = 0.0

            # Publish the pose
            mean, cov = self.get_pose()
            print(mean)
            self.mean = np.vstack([self.mean, mean])
            self.plot()
            # print(cov)
            # self.writer.writerow([float(sensor_data[0]), float(sensor_data[1]), float(sensor_data[2]), float(sensor_data[3]), float(sensor_data[4]), float(sensor_data[5]), float(sensor_data[6])])
            self.writer.writerow([float(sensor_data[0]), mean[0], mean[1], mean[2], cov[0][0], cov[0][1], cov[0][2], cov[1][0], cov[1][1], cov[1][2], cov[2][0], cov[2][1], cov[2][2]])
            # self.mean = np.vstack([self.mean, mean])
        self.prev_time = float(sensor_data[0])

    def end(self):
        self.f.close()

if __name__ == "__main__":
    pfLoc = PFLocalization()
    # read ground truth data
    with open("data/continuous_gt_1.csv", mode ='r', newline='') as file:
        csvFile = csv.reader(file)
        for lines in csvFile:
            pfLoc.gt_x = np.append(pfLoc.gt_x, float(lines[1]))
            pfLoc.gt_y = np.append(pfLoc.gt_y, float(lines[2]))
    pfLoc.gt_x_actual = pfLoc.gt_x
    pfLoc.gt_y_actual = pfLoc.gt_y

    # obtain trajectary area
    pfLoc.leftX = min(pfLoc.gt_x)
    pfLoc.rightX = max(pfLoc.gt_x)
    pfLoc.leftY = min(pfLoc.gt_y)
    pfLoc.rightY = max(pfLoc.gt_y)

    # predicted mean for plotting
    with open("data/predict_real.csv", mode ='r', newline='') as file:
        csvFile = csv.reader(file)
        for lines in csvFile:
            mean = np.array([float(lines[1]), float(lines[2]), float(lines[3])])
            pfLoc.predicted_mean = np.vstack([pfLoc.predicted_mean, mean])

    if len(sys.argv) > 1:
        pfLoc.real_time = True
        with open(sys.argv[1], mode ='r', newline='') as file:
            csvFile = csv.reader(file)
            time = 0.0
            for lines in csvFile:
                sensor_data = np.array([time,
                                        float(lines[1]), float(lines[2]), float(lines[3]),
                                        float(lines[4]), float(lines[5]), float(lines[6])])
                time = time + 100
                pfLoc.run(sensor_data)
            pfLoc.final_plot()
                
    else:
        pfLoc.real_time = True
        HOST = '35.0.131.136'
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