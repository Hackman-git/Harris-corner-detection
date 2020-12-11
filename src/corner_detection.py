import cv2 as cv
import numpy as np
import numpy.linalg as lin
import pandas as pd


def read_image(path):
    img = cv.imread(path)
    print("image shape: ", img.shape)
    return img


path1 = r"C:\Users\abdul\Desktop\CS 512 - Computer vision\Homework\HW2\data\home.jpg"
path2 = r"C:\Users\abdul\Desktop\CS 512 - Computer vision\Homework\HW2\data\home_rot.jpg"

img = read_image(path1)
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img2 = read_image(path2)
img2_gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)


def pad_amount(window_size, img):
    '''
    We assume that window_size is square
    e.g 4x4
    '''
    size = img.shape
    r = size[1]
    right = r % window_size
    if right != 0:
        if (window_size - right) % 2 != 0:
            right_pad = int((window_size - right) / 2)
            left_pad = (window_size - right) - right_pad
        else:
            right_pad = (window_size - right) / 2
            left_pad = right_pad
    else:
        left_pad, right_pad = (0, 0)

    t = size[0]
    top = t % window_size
    if top != 0:
        if (window_size - top) % 2 != 0:
            top_pad = int((window_size - top) / 2)
            bottom_pad = (window_size - top) - top_pad
        else:
            top_pad = (window_size - top) / 2
            bottom_pad = top_pad
    else:
        bottom_pad, top_pad = (0, 0)

    return ([int(left_pad), int(right_pad), int(top_pad), int(bottom_pad)])


def corners(window_size, img_gray, sigma=0.0, k=0.04, threshold=0.01):
    '''
    :param window_size:
    :param img_grad_x:
    :param img_grad_y:
    :return:
    '''
    img_grad_x = cv.Sobel(img_gray, cv.CV_64F, 1, 0)
    img_grad_y = cv.Sobel(img_gray, cv.CV_64F, 0, 1)
    img_grad_x = cv.GaussianBlur(img_grad_x, (3, 3), sigma)
    img_grad_y = cv.GaussianBlur(img_grad_y, (3, 3), sigma)

    # pad the image so the windows fit at the edges
    left_pad, right_pad, top_pad, bottom_pad = pad_amount(window_size, img_grad_x)
    img_grad_x_pd = cv.copyMakeBorder(img_grad_x, top_pad, bottom_pad, left_pad, right_pad, cv.BORDER_CONSTANT)
    img_grad_y_pd = cv.copyMakeBorder(img_grad_y, top_pad, bottom_pad, left_pad, right_pad, cv.BORDER_CONSTANT)
    print('padded image shape: ', img_grad_x_pd.shape)
    rows, cols = img_grad_x_pd.shape
    # The windows won't overlap so we should have size of patches equal to (rows/window_size) * (cols/window_size)
    # store the correlation matrix for each window here
    patches = []
    # store the correlation matrix for each point in a window here
    g = []
    data = pd.DataFrame(columns=['window', 'corner_measure', 'y1', 'y2', 'x1', 'x2'])

    # iterate over the whole image with windows
    for row in range(0, rows, window_size):
        for col in range(0, cols, window_size):
            x_patch = img_grad_x_pd[row:row + window_size, col:col + window_size]
            y_patch = img_grad_y_pd[row:row + window_size, col:col + window_size]

            # iterate over each pixel in a window and compute gradient product
            for i in range(x_patch.shape[0]):
                for j in range(x_patch.shape[1]):
                    grad = np.array([x_patch[i, j], y_patch[i, j]])
                    grad.shape = (2, 1)
                    g.append(np.matmul(grad, grad.T))

            temp = np.zeros(shape=(2, 2))
            # obtain correlation matrix of a window by summing up all gradient products in g
            for l in g:
                temp = np.add(temp, l)
            # store the correlation matrix in patches list
            patches.append(temp)
            g = []

            det = lin.det(temp)
            tr = k * (np.trace(temp) ** 2)
            c_measure = det - tr

            data.loc[data.shape[0] + 1] = [len(patches), c_measure, row, row + window_size - 1, col,
                                           col + window_size - 1]

    # better localization of corners
    data = data.sort_values('corner_measure', ascending=False).reset_index(drop=True)
    # take top windows according to Harris threshold
    data = data.iloc[:int(data.shape[0] * threshold), :]
    print('data for windows:\n', data)
    data2 = pd.DataFrame(columns=['x', 'y', 'corner_measure'])
    # iterate over top windows
    for i in range(data.shape[0]):
        # initialize max of cornerness measure over all window points
        m = -100000000000000000
        # initialize location of corner
        curr_max_loc = (0, 0)
        x1, x2, y1, y2 = data.loc[i, 'x1'], data.loc[i, 'x2'], data.loc[i, 'y1'], data.loc[i, 'y2']
        # calculate cornerness in each point
        for l in range(int(y1), int(y2)):
            for j in range(int(x1), int(x2)):
                grad = np.array([img_grad_x_pd[l, j], img_grad_y_pd[l, j]])
                grad.shape = (2, 1)
                corr_mat = np.matmul(grad, grad.T)
                det = lin.det(corr_mat)
                tr = k * (np.trace(corr_mat) ** 2)
                c_measure = det - tr

                # update max cornerness
                if c_measure > m:
                    m = c_measure
                    curr_max_loc = (l, j)
        # save results in dataframe
        data2.loc[data2.shape[0] + 1] = [curr_max_loc[1], curr_max_loc[0], m]

    return (data, data2, (left_pad, top_pad))


# data = data.sort_values('corner_measure', ascending=False).head(100)
# print(data)

# for i in range(data.shape[0]):
#     temp = data.iloc[i,:]
#
#     cv.rectangle(img, (int(temp['x1']-left_pad), int(temp['y1']-top_pad)),
#                  (int(temp['x2']-left_pad),int(temp['y2']-top_pad)), (255,0,0), 2)
#
# cv.imshow('img', img)
# cv.waitKey(0)
# cv.destroyAllWindows()


def show_corners(window_size, img_grayscale, img_original, sigma=0.0, k=0.04, threshold=0.01):
    data, data2, _ = corners(window_size, img_grayscale, sigma, k, threshold)
    print('Corner coordinates:\n', data2)
    for i in range(data2.shape[0]):
        temp = data2.iloc[i, :]
        pad = 2
        cv.rectangle(img_original, (int(temp['x'] - pad), int(temp['y'] - pad)),
                     (int(temp['x'] + pad), int(temp['y'] + pad)), (255, 0, 0), 2)

    return img_original


def match_corners(img1, img2, num_points=20):
    sift = cv.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    matcher = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    out = cv.drawMatches(img_gray, kp1, img2_gray, kp2, matches[:num_points], None)

    return (out)


def run():
    flag = True
    while flag:
        print("\nEnter the following params separated by a space:\n"
              "window size, sigma, k, threshold. e.g.\n"
              "6 1 0.04 0.01\n\nUse these ranges as a guide:\n"
              "window size [4,6,8]\nsigma [0,1,2,3]\nk[0.01, 0.04, 0.06, 0.1]\nthreshold [0.01 - 0.05]\n"
              "Note that threshold means top x% of sorted corner points identified\n computation may "
              "take a short while...\n")
        params = input().split()
        window_size, sigma, k, threshold = int(params[0]), float(params[1]), float(params[2]), float(params[3])

        out1 = show_corners(window_size, img_gray, img, sigma, k, threshold)
        cv.imshow('image1 corners', out1)
        print('\npress enter key to view next image corners or q to exit program')
        key = cv.waitKey(0)
        if key == ord('q'):
            cv.destroyAllWindows()
            break
        cv.destroyAllWindows()
        out2 = show_corners(window_size, img2_gray, img2, sigma, k, threshold)
        cv.imshow('image2 corners', out2)
        print('\npress enter key to proceed to feature matching. Press ESC to go back to start '
              'menu or q to quit program')
        key = cv.waitKey(0)
        if key == 27:
            cv.destroyAllWindows()
            continue
        elif key == ord('q'):
            cv.destroyAllWindows()
            break
        else:
            cv.destroyAllWindows()
            print('\nenter the number of points to match in both images and hit enter: ')
            num = int(input())
            out3 = match_corners(img, img2, num)

        cv.imshow('feature match', out3)
        print('\npress enter key to return to start menu')
        cv.waitKey(0)
        cv.destroyAllWindows()


if __name__ == '__main__':
    run()
    # d = cv.cornerHarris(img_gray,4,3,0.04)
    #
    # d = cv.dilate(d, None)
    # img[d>0.03*d.max()]=[255,0,0]
    #
    # cv.imshow('d',img)
    # if cv.waitKey(0) & 0xff == 27:
    #     cv.destroyAllWindows()