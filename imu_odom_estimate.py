import socket
import math
import numpy as np

class Odometry:
    def __init__(self):
        self.x = 0.0    # x position in meters
        self.y = 0.0    # y position in meters
        self.theta = 0.0    # orientation in radians
        self.gyro_z=0.0

        self.calibrate_cnt = 0
        self.sum = 0
        self.calibration = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        # IMU parameters
        self.last_timestamp = None
        self.alpha = 0.5    # complementary filter coefficient

    def calibrate(self, imu_data):
        self.sum = self.sum + imu_data
        self.calibrate_cnt += 1
        if self.calibrate_cnt >= 300:
            self.calibration = self.sum/self.calibrate_cnt
            print("calibration done")
            return 1
        return 0

    def update(self, timestamp, imu_data):
        imu_data = imu_data - self.calibration
        acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z = imu_data

        dt = 0.01
        self.gyro_z += gyro_z * dt
        self.theta = self.alpha * (self.theta + gyro_z * dt) + (1 - self.alpha) * math.atan2(math.sin(self.theta), math.cos(self.theta))

        # calculate displacement
        dx = (acc_x * math.cos(self.theta) + acc_y * math.sin(self.theta)) * dt
        dy = (-acc_x * math.sin(self.theta) + acc_y * math.cos(self.theta)) * dt

        # update position
        self.x += dx
        self.y += dy

    def get_pose(self):
        return (self.x, self.y, self.theta)

HOST = '10.0.0.115'  # IP address of the device running HyperIMU
PORT = 2055           # Port number used in HyperIMU

odometry = Odometry()

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    print(f"Listening for connections on {HOST}:{PORT}...")
    conn, addr = s.accept()
    with conn:
        print('Connected by', addr)
        done = 0
        print("Please hold still. Calibration in progress...")
        while True:
            data = conn.recv(1024)
            if not data:
                break
            # print(f"Received data: {data}")
            try:
                # sample data format: "timestamp,gyro_x,gyro_y,gyro_z,acc_x,acc_y,acc_z"
                timestamp, gyro_x, gyro_y, gyro_z, acc_x, acc_y, acc_z = map(float, data.decode().split(','))
                
                if not done:
                    done = odometry.calibrate(np.array([acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]))
                    continue

                odometry.update(timestamp, np.array([acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]))

                # get current position and send back as ACK
                position = odometry.get_pose()
                print("{}, {}, {}".format(position[0], position[1], position[2]))
                # print("acc -> x: {}, y: {}, z: {}".format(acc_x, acc_y, acc_z))
                # print("Gyro -> x: {}, y: {}, z: {}".format(gyro_x, gyro_y, gyro_z))
                conn.sendall(f"ACK,{position[0]},{position[1]},{position[2]}".encode())

            except ValueError:
                pass