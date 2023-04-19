import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from sklearn.mixture import GaussianMixture
import csv

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
    shift_dist_max = min((max(x) - min(x)) / 20, (max(y) - min(y)) / 20)
    shift_dist_min = min((max(x) - min(x)) / 40, (max(y) - min(y)) / 40)

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

    #  total interpolation x, y, z and local maximas
    whole_xs = []
    whole_ys = []
    whole_zs = []
    whole_local_max_numbs = []

    for i in range(len(x)):
        global_max = [x[i], y[i]]
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

        # # plot
        # # plt.figure(figsize=(16,6))
        # plt.figure(figsize=(8, 6))
        # # plt.subplot(121)
        # plt.contourf(interp_xi, interp_yi, interp_zi, levels=15, cmap='GnBu')
        # plt.plot(groundtruthX, groundtruthY, label='Ground Truth', color='darkgreen')
        # plt.plot(groundtruthX, groundtruthY,'x', color='cyan')
        # plt.colorbar(label='Possibility')
        # plt.xlabel('X Coordinate')
        # plt.ylabel('Y Coordinate')
        # plt.title(f'Possibility Distribution')
        # plt.legend()

        whole_xs.append(interp_xi)
        whole_ys.append(interp_yi)
        whole_zs.append(interp_zi)
        whole_local_max_numbs.append(local_max_num)

    return whole_xs, whole_ys, whole_zs, whole_local_max_numbs
    
# Gaussian Mixture Model (GMM) extraction -- return each gaussian model's average coordinate and covariance
def gmm_extr(inter_xi, inter_yi, inter_zi, local_max_num, threshold = 0.48):
    multi_model = remove_low_possibility(inter_xi, inter_yi, inter_zi, threshold)
    multi_model = np.array(multi_model)

    # apply GMM to get predicted label -- for plot
    label_pred = GaussianMixture(n_components=local_max_num).fit_predict(multi_model)

    # apply GMM to get each model's average & covariance
    gm = GaussianMixture(n_components=local_max_num, random_state=0).fit(multi_model)
    print("Each average located at: \n")
    print(gm.means_)
    print("\nEach corresponding covariance: \n")
    print(gm.covariances_)

    # # plot
    # plt.figure(figsize=(6.4, 6))
    # # plt.subplot(122)
    # plt.xlim(min(inter_xi), max(inter_xi))
    # plt.ylim(min(inter_yi), max(inter_yi))
    # plt.scatter(multi_model[:, 0], multi_model[:, 1], marker=',', c=label_pred)
    # plt.scatter(np.array(gm.means_)[:, 0], np.array(gm.means_)[:, 1], marker='x')
    # # plt.colorbar()
    # plt.xlabel('X Coordinate')
    # plt.ylabel('Y Coordinate')
    # plt.legend()
    # plt.title("GMM Classified Gaussian Mixture Models")
    # plt.show()

    return gm.means_, gm.covariances_   # one to one corresponding

# remove all the low possibility points and only get points coordinates that possibility higher than the threshold
def remove_low_possibility(inter_xi, inter_yi, inter_zi, threshold):
    coord = []
    for index_yi in range(len(inter_yi)):
        for index_xi in range(len(inter_xi)):
            if inter_zi[index_yi][index_xi] >= threshold:
                coord.append([inter_xi[index_xi], inter_yi[index_yi]])

    return coord

# Given IMU estimated location, combine with the RSSI possibility GMM to obtain corrected location
rssi_cts = []
def rssi_correct_imu(imu_es, imu_cov, multi_model_means, multi_model_covs):
    current_min_dist = 1e7
    for i in range(len(multi_model_means)):
        dist = np.linalg.norm([imu_es[0] - multi_model_means[i][0], imu_es[1] - multi_model_means[i][1]])
        if dist < current_min_dist:
            current_min_dist = dist
            rssi_ct = list(multi_model_means[i])  #for plot
            nearest_model_center = np.array(multi_model_means[i]).reshape(2, 1)
            nearest_model_cov = np.array(multi_model_covs[i])
    
    rssi_cts.append(rssi_ct)   #global variable
    print(rssi_cts)
    # calculate corrected location using center and cov
    imu_es = imu_es.reshape(2, 1)
    kal_gain = imu_cov @ np.linalg.inv(imu_cov + nearest_model_cov)
    corrected_pos = imu_es + 0.4 * kal_gain @ (nearest_model_center - imu_es)

    print(corrected_pos)

    return corrected_pos, rssi_ct

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
    # read the ground truth and imu data
    gt_file = 'continuous_gt_1.csv'
    imu_file = 'continuous_imu_pf_1.csv'
    pf_file = 'particle_filter_pose_1.csv'

    trX = []    # ground truth
    trY = []
    trTH = []

    imu_es_centers = []  # imu data needs to be imported
    imu_es_covs = []

    pf_es_centers_x = []  # particle filter estimate
    pf_es_centers_y = []

    # read ground truth
    with open(gt_file, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            trX.append(float(row[0]))
            trY.append(float(row[1]))
            trTH.append(float(row[2]))
            print('\n')
            print(row[1], row[2])

    # read imu data
    with open(imu_file, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            imu_es_centers.append([float(row[1]), float(row[2]), float(row[3])])

            temp_cov = []
            for i in range(9):
                temp_cov.append(float(row[i + 4]))

            temp_cov = np.array(temp_cov).reshape(3, 3)
            imu_es_covs.append(temp_cov)
            print('\n')
            print(row[1], row[2])
        imu_es_centers = np.array(imu_es_centers)
        # imu_es_covs = np.array(imu_es_covs)

    whole_x, whole_y, whole_z, whole_local_max_nums = simulate_pdf(trX, trY)

    # load particle filter data
    with open(pf_file, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            pf_es_centers_x.append(float(row[1]))
            pf_es_centers_y.append(float(row[2]))
            print('\n')
            print(row[1], row[2])

    corrected_poses = [np.array([imu_es_centers[0][0], imu_es_centers[0][1]]).reshape(2, 1)]
    rssi_cts = []

    for i in range(1, len(whole_x)):
        next_center = np.array(corrected_poses[i - 1]).reshape(2, 1) + np.array([imu_es_centers[i][0] - imu_es_centers[i - 1][0], imu_es_centers[i][1] - imu_es_centers[i - 1][1]]).reshape(2, 1)

        print(np.array(corrected_poses[i - 1]))
        print(np.array([imu_es_centers[i][0] - imu_es_centers[i - 1][0], imu_es_centers[i][1] - imu_es_centers[i - 1][1]]))

        wifi_centers, wifi_covs = gmm_extr(whole_x[i], whole_y[i], whole_z[i], whole_local_max_nums[i], threshold=0.55)
        corrected_pos, c_ct = rssi_correct_imu(next_center, imu_es_covs[i][:2, :2], wifi_centers, wifi_covs)
        # corrected_pos, c_ct = rssi_correct_imu(np.array([trX[i], trY[i]]).reshape(2, 1), imu_es_covs[i][:2, :2], wifi_centers, wifi_covs)
        
        corrected_poses.append(corrected_pos)
        print("now: " + str(i) + ", " + str(len(whole_x)))

    corrected_poses = np.array(corrected_poses)

    # calculate RMSE
    mse_imu = 0
    mse_kf = 0
    mse_pf = 0

    for i in range(len(corrected_poses)):
        mse_imu += np.square(np.linalg.norm([trX[i] - imu_es_centers[i][0], trY[i] - imu_es_centers[i][1]])) / len(trX)
        mse_kf += np.square(np.linalg.norm([trX[i] - corrected_poses[i][0], trY[i] - corrected_poses[i][1]])) / len(trX)
        mse_pf += np.square(np.linalg.norm([trX[i] - pf_es_centers_x[i], trY[i] - pf_es_centers_y[i]])) / len(trX)

    rmse_imu = np.power(mse_imu, 0.5)
    rmse_kf = np.power(mse_kf, 0.5)
    rmse_pf = np.power(mse_pf, 0.5)
    print("RMSE of IMU: " + str(rmse_imu))
    print("RMSE of Kalman Filter: " + str(rmse_kf))
    print("RMSE of Particle Filter: " + str(rmse_pf))

    # plot
    rssi_cts = np.array(rssi_cts)
    plt.figure()
    plt.plot(trX, trY, label='Ground Truth')
    plt.plot(imu_es_centers[:, 0], imu_es_centers[:, 1], label='IMU Estimation')
    plt.plot(corrected_poses[:, 0], corrected_poses[:, 1], label='Kalman Filter Estimation')
    plt.plot(pf_es_centers_x, pf_es_centers_y, label='Particle Filter Estimation')
    # plt.scatter(rssi_cts[:, 0], rssi_cts[:, 1], marker='x')
    # plt.scatter(imu_es_centers[:, 0], imu_es_centers[:, 1], marker='x', color='red')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    # plt.title("")
    plt.legend()
    plt.show()


    # # For testing the RSSI IMU correction
    # imu_es_center = [40.99702809, 39.98111026, -0.15718418]
    # imu_es_cov = np.array([[2.59943093e-01, 1.49086640e-02, 2.16506722e-04], 
    #             [1.49086640e-02, 2.52513672e-01, -1.85946646e-03], 
    #             [2.16506722e-04, -1.85946646e-03, 3.81160943e-03]])
    
    # centers = [[42.13483032, 48.58744849], 
    #            [75.34173564, 23.3605504], 
    #            [17.52775467, 72.00524633], 
    #            [60.16556563, 20.96756063]]
    # covs = [[[1.40474574e+01, -5.68109855e+00], 
    #         [-5.68109855e+00, 9.55252954e+00]], 

    #         [[ 1.00000000e-06, 2.12045811e-27], 
    #         [ 2.12045811e-27, 4.48950761e-01]], 

    #         [[ 1.25342215e+00, 1.26543986e-14], 
    #         [ 1.26760799e-14, 1.72396808e+00]], 

    #         [[ 2.31286145e+00, -2.56731084e-16], 
    #         [-2.78817369e-16, 1.90574692e+00]]]
    
    # rssi_correct_imu(np.array([imu_es_center[0], imu_es_center[1]]).reshape(2, 1), imu_es_cov[:2, :2], centers, covs)