import time
import csv
import numpy as np

# open the file in the write mode
f = open('data/simulated_imu.csv', 'w', newline='')
writer = csv.writer(f)
writer.writerow([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
def turn(right):
    total_elapsed = 0.1
    while total_elapsed < 1.0:
        start_time = time.time() # Get the current time before processing
        rotate = np.pi/2
        if (right):
            rotate = -np.pi/2
        writer.writerow([time.time(), 0.0, 0.0, 0.0, 0.0, 0.0, rotate])
        elapsed_time = time.time() - start_time # Calculate the time taken for processing
        sleep_time = max(0.0, 0.1 - elapsed_time) # Calculate the time to sleep for
        total_elapsed = total_elapsed + 0.1

        time.sleep(sleep_time) # Sleep for the remaining time to achieve 10Hz

def straight(meters):
    total_elapsed = 0.1
    while total_elapsed < meters*1.0:
        start_time = time.time() # Get the current time before processing
        writer.writerow([time.time(), 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        elapsed_time = time.time() - start_time # Calculate the time taken for processing
        sleep_time = max(0.0, 0.1 - elapsed_time) # Calculate the time to sleep for
        total_elapsed = total_elapsed + 0.1

        time.sleep(sleep_time) # Sleep for the remaining time to achieve 10Hz

# start at 0, 0 assume facing towards +X
# turn to -Y -> -pi/2
turn(1)
straight(6)

turn(0)
straight(1)
turn(0)
straight(6)

turn(1)
straight(1)
turn(1)
straight(6)

turn(0)
straight(1)
turn(0)
straight(6)

turn(1)
straight(1)
turn(1)
straight(6)

turn(0)
straight(1)
turn(0)
straight(6)