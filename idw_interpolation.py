import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import simulate_imu

def idw_interpolation(x, y, xi, yi, avg_signal_strengths, p=2):
    """
    Return zi 2D list value after applied IDW interpolation method. More existing points will have better performance.
    Input: x, y are existing points; xi, yi are interpolate points;
        avg_signal_strengths is RSSI value for existing points;
        p is the power parameter (positive real number), typically 0.5~3, generally being 2.
    Authors: Haochi P.
    """
    zi = []
    for i in range(len(yi)):    #for each interpolate point
        sub_zi = []
        for k in range(len(xi)):
            weight = []
            break_flag = False

            for j in range(len(x)):     #calculate weight from each existing point
                dist = np.sqrt(np.square(xi[k] - x[j]) + np.square(yi[i] - y[j]))

                if dist < 1e-7:     #overlap with existing point
                    sub_zi.append(avg_signal_strengths[j])
                    break_flag = True
                    break
                else:
                    weight.append(np.power(dist, -p))
                
            if break_flag == True:
                continue
            else:
                sub_zi.append(np.sum(np.multiply(weight, avg_signal_strengths)) / np.sum(weight))

        zi.append(sub_zi)

    return zi

# slightly shift from given ground truth
def groundtruth_shift(x, y):
    shift_dist_max = min((max(x) - min(x)) / 15, (max(y) - min(y)) / 15)
    shift_dist_min = min((max(x) - min(x)) / 30, (max(y) - min(y)) / 30)

    shifted_x = []
    shifted_y = []
    for i in range(len(x)):
        scale_x = 0
        scale_y = 0
        while scale_x == 0:
            scale_x = random.randint(-1, 1)
        while scale_y == 0:
            scale_y = random.randint(-1, 1)

        shifted_x.append(x[i] + scale_x * random.uniform(shift_dist_min, shift_dist_max))
        shifted_y.append(y[i] + scale_y * random.uniform(shift_dist_min, shift_dist_max))

    return shifted_x, shifted_y

# simulate multiple RSSI layer distribution matching belief
def simulate_pdf(x, y):
    # shift ground truth
    groundtruthX = x    # recording the groundtruth
    groundtruthY = y
    x, y = groundtruth_shift(x, y)

    # obtain trajectary area
    leftX = min(x)
    rightX = max(x)
    leftY = min(y)
    rightY = max(y)

    for i in range(len(x)):
        global_max = [x[i], y[i]]
        local_max_num = random.randint(3, 7)
        existing_points_num = random.randint(20, 30)

        spead_dorder = 5        #local maxima may outside of map border
        dist = 8        #local maxima should at least away from global one this distance

        existing_points = []    # x, y, possibility

        global_max.append(random.uniform(0.85, 0.91))       #MAX_POSSIBILITY
        existing_points.append(global_max)
        
        for j in range(local_max_num + existing_points_num):
            # create points position
            temp_dist = 0
            while temp_dist < dist:
                xi = random.uniform(leftX - spead_dorder, rightX + spead_dorder)
                yi = random.uniform(leftY - spead_dorder, rightY + spead_dorder)
                temp_dist = np.linalg.norm([xi - global_max[0], yi - global_max[1]])

            # assign possibility for each existing points
            if j < local_max_num:   # local maxima
                existing_points.append([xi, yi, random.uniform(0.5, 0.69)])
            elif j < (local_max_num + int(existing_points_num / 4)):    # small group of some possibilities
                existing_points.append([xi, yi, random.uniform(0.20, 0.35)])
            else:   #other possibilities
                existing_points.append([xi, yi, random.uniform(0.04, 0.17)])

        # print(existing_points)

        # interpolation
        interp_xi = np.linspace(leftX - spead_dorder, rightX + spead_dorder, 50)
        interp_yi = np.linspace(leftY - spead_dorder, rightY + spead_dorder, 50)
        zi1, zi2, zi3 = list(zip(*([(q[0], q[1], q[2]) for q in existing_points])))
        # print(zi1)

        interp_zi = idw_interpolation(zi1, zi2, interp_xi, interp_yi, zi3, 2)
        # interp_zi = griddata((existing_points[:][0], existing_points[:][1]), existing_points[:][2], (interp_xi, interp_yi), method='cubic')

        # plot
        plt.figure()
        plt.contourf(interp_xi, interp_yi, interp_zi, levels=15, cmap='inferno')
        plt.plot(groundtruthX, groundtruthY)
        plt.plot(groundtruthX, groundtruthY,'x')
        plt.colorbar(label='Possibility')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title(f'Possibility Distribution')
        plt.legend()
        plt.show()

# from IMU turning (left/right) direction to global coordinate direction
def head_to_next(current_head_to, turn):
    next_head_to = []

    if current_head_to == [1, 0]:
        if turn == 0:   # left
            next_head_to = [0, 1]
        elif turn == 1: # right
            next_head_to = [0, -1]

    elif current_head_to == [-1, 0]:
        if turn == 0:
            next_head_to = [0, -1]
        elif turn == 1:
            next_head_to = [0, 1]

    elif current_head_to == [0, 1]:
        if turn == 0:
            next_head_to = [-1, 0]
        elif turn == 1:
            next_head_to = [1, 0]

    elif current_head_to == [0, -1]:
        if turn == 0:
            next_head_to = [1, 0]
        elif turn == 1:
            next_head_to = [-1, 0]

    return next_head_to
        
# transfer IMU control command to actual trajectory
def imu_control_coordinate_transfer(turns, dists):
    tr_x = [0]
    tr_y = [0]
    current_pos_x = 0
    current_pos_y = 0
    current_head_to = [1, 0]

    for i in range(len(turns)):
        # turn: 1 right / 0 left .. starting from pointing to +x .. turn pi/2
        # # simulate_imu
        # simulate_imu.turn(turns[i])
        # simulate_imu.straight(dists[i])

        # transfer into x-y plane coordinate
        current_head_to = head_to_next(current_head_to, turns[i])
        current_pos_x += current_head_to[0] * dists[i]
        current_pos_y += current_head_to[1] * dists[i]
        tr_x.append(current_pos_x)
        tr_y.append(current_pos_y)

    return tr_x, tr_y

# particle filter update and resample step
def particle_update(parti_x, parti_y, possi_x, possi_y, possi_z, parti_weights): # input [x, y], first time wight[] is uniform
    likelihoods = []
    for i in range(len(parti_x)):
        # find surround points
        for j in range(len(possi_x)):
            if possi_x[j] > parti_x[i]:
                border_x_r = j
                break
        for k in range(len(possi_y)):
            if possi_y[k] > parti_y[i]:
                border_y_r = k
                break
        
        right_x = possi_x[border_x_r]
        right_y = possi_y[border_y_r]
        left_x = possi_x[border_x_r - 1]
        left_y = possi_y[border_y_r - 1]

        s_points = [[left_x, left_y, possi_z[border_x_r - 1, border_y_r - 1]], [left_x, right_y, possi_z[border_x_r - 1, border_y_r]], [right_x, left_y, possi_z[border_x_r, border_y_r - 1]], [right_x, right_y, possi_z[border_x_r, border_y_r]]]

        # interpolation - get current particle possibility value: parti_z
        idw_weight = []
        for s_point in s_points:
            dist = np.sqrt(np.square(s_point[0] - parti_x[i]) + np.square(s_point[1] - parti_y[i]))

            if dist < 1e-7:
                parti_z = s_point[2]
                break_flag = True
                break
            else:
                idw_weight.append(np.power(dist, -2))
                
        if break_flag == False:
            possibility_value = []
            for s_point in s_points:
                possibility_value.append(s_point[2])
            parti_z = np.sum(np.multiply(idw_weight, possibility_value)) / np.sum(idw_weight)

        likelihoods.append(parti_z)

    # update step
    parti_weights *= likelihoods / np.sum(likelihoods)

    # resample
    indices = np.random.choice(len(parti_x), size=len(parti_x), p=parti_weights)
    parti_x = parti_x[indices]
    parti_y = parti_y[indices]
    parti_weights = np.ones_like(parti_weights) / len(parti_weights)

    return parti_x, parti_y, parti_weights
        
        

if __name__ == "__main__":     
    trX = [0, 15, 20, 27, 37, 44]
    trY = [0, 8, 13, 30, 27, 15]

    # trX, trY = groundtruth_shift(trX, trY)
    # print(str(trX) + "\n" + str(trY))

    simulate_pdf(trX, trY)


