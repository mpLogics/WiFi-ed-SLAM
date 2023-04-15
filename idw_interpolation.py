import numpy as np

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


