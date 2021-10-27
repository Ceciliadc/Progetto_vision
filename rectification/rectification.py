import os
import re
import cv2
import numpy as np
import glob


def atof(text):
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval


def natural_keys(text):
    return [atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text)]


def order_points(pts):
    xSorted = pts[np.argsort(pts[:, 0]), :]

    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    # if use Euclidean distance, it will run in error when the object
    # is trapezoid. So we should use the same simple y-coordinates order method.

    # now, sort the right-most coordinates according to their
    # y-coordinates so we can grab the top-right and bottom-right
    # points, respectively
    rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
    (tr, br) = rightMost

    return np.array([tl, tr, br, bl], dtype="int")


def get_corners(image, alpha, beta, t):

    rect = None
    new_image = np.zeros(image.shape, image.dtype)
    new_image[:, :, :] = np.clip(alpha * image[:, :, :] + beta, 0, 255)
    gray = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 9, 75, 75)

    _, thresh = cv2.threshold(
        blur, t, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(
        thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    try:
        c = max(contours, key=cv2.contourArea)
    except:
        return rect, False

    x, y, w, h = cv2.boundingRect(c)
    c_size = h * w
    accuracy = 0.09*cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, accuracy, True)
    area = cv2.contourArea(approx)
    corners_img = image.copy()

    if len(approx) == 4:

        if area > 10000 and abs(c_size - image.shape[0] * image.shape[1]) >= 0.5:
            cv2.drawContours(corners_img, [approx], 0, (0, 255, 0), 2)

            approx = np.reshape(approx, (4, 2))
            rect = order_points(approx)

            for i, corner in enumerate(rect):
                x_corner = int(corner[0])
                y_corner = int(corner[1])

                cv2.circle(corners_img, (x_corner, y_corner), 1, (255, 0, 0), 2)
                cv2.putText(corners_img, f'{i}', (x_corner, y_corner),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, color=(255, 0, 0))

            return rect, True
    return rect, False


def warp_image(src_points, crop_img):
    (tl, tr, br, bl) = src_points

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="int")
    ret, _ = cv2.findHomography(src, dst, cv2.RANSAC)
    warped_image = cv2.warpPerspective(crop_img, ret, (maxWidth, maxHeight))

    return warped_image


im_path = "../rectification/painting"
im_list = glob.glob(f'{im_path}/*.png', recursive=False)
output_path = "../rectification/painting_rect1"

try:
    os.makedirs(output_path)
except:
    print('error: cannot build', output_path)

im_list.sort(key=natural_keys)

for image in im_list:
    im_name = os.path.splitext(os.path.basename(image))[0]

    try:
        file = open(im_path + '/' + im_name + ".txt", 'r')
    except:
        continue
    txt = file.readlines()

    im = cv2.imread(image)

    if im is None:
        continue

    im_h, im_w, _ = im.shape

    for i in range(len(txt)): # per ogni riga del file, quindi per ogni bounding box

        coordinates = txt[i].split()
        if coordinates[0] == '0': #considero solo i quadri, quindi id=0

            # prendo le coordinate dal file txt e le rendo non normalizzate
            ncx, ncy, nw, nh = float(coordinates[1]), float(coordinates[2]), float(coordinates[3]), float(coordinates[4])

            w = nw * im_w
            h = nh * im_h

            cx = ncx * im_w
            cy = ncy * im_h

            # calcolo l'angolo in alto a sinistra
            x1 = int(cx - (float(w) / 2))
            y1 = int(cy - (float(h) / 2))
            x2 = int(x1 + float(w))
            y2 = int(y1 + float(h))

            # taglio l'immagine tenendo solo la bounding box
            crop = im[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)]

            found = False
            j = 0
            params = [[1.0, 0, 80], [2.0, -200, 80],  [4.0, -450, 150], [4.0, -180, 250], [3.0, -250, 90],
                        [5.0, -500, 100], [10.0, -500, 70], [5.0, -500, 50], [1.0, 0, 100], [4.0, -200, 80], [4.0, -200, 150]]
            params_old = [[10.0, -500, 0.04, 70], [1.0, 0, 0.03, 100], [7.0, -500, 0.08, ], [1.0, 0, 0.07],
                      [10.0, -300, 0.02], [2.0, 0, 0.01], [7.0, -500, 0.09], [1.0, 0, 0.06], [1.0, 0, 0.05], [7.0, -800, 0.045]]

            crop_h, crop_w = crop.shape[0], crop.shape[1]
            crop = cv2.resize(crop, (crop_w//4, crop_h//4))
            src = []

            while not found and j != 11:
                src, found = get_corners(crop, params[j][0], params[j][1], params[j][2])
                j += 1

            try:
                warped = warp_image(src, crop)
            except:
                continue

            if cv2.countNonZero(cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY)) != 0:
                cv2.imwrite(
                    f'{output_path}/{im_name}-rect{i}.png', warped)
            else:
                continue
