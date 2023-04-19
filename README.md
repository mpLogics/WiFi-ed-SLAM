# WiFi-ed SLAM

WiFi - ed(enhanced) SLAM aims at incorporate signal strength, time of flight and orientation information from WiFi and Bluetooth signals along with IMU data to enhance the accuracy and precision of conventional localization and mapping approaches by adding in extra dimensions of information. 

Furthermore, we also aim at creating a map of the simulated raw field strength and identify spatial features to aid in loop-closure.


simulate_imu.py -> This file can be run to simulate the IMU data for a given path. The IMU data will be saved as simulated_imu.csv file.

imu_odom_loc_pf.py -> This file has the implementation of our Particle Filter this can take the simulated data as input to predict the state. But, It requires a ground truth file to simulate WiFI to update the prediction.

