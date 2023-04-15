import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

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

if __name__ == "__main__":     
    trX = [0, 15, 20, 27, 37, 44]
    trY = [0, 8, 13, 30, 27, 15]

    # trX, trY = groundtruth_shift(trX, trY)
    # print(str(trX) + "\n" + str(trY))

    simulate_pdf(trX, trY)


