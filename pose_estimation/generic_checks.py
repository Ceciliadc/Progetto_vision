import cv2
import numpy as np


# returns the position of the box of the same person in case she is not mooving
def check(bbox1, labels_2):
    file = open(labels_2)
    bbox2 = file.readlines()

    for bbox in bbox2:
        if check_same_person(bbox1, bbox):
            file.close()
            return bbox2.index(bbox)
    file.close()
    return -1


def check_same_person(bbox1, bbox2):
    x1, y1, w1, h1 = list(map(float, bbox1))
    x2, y2, w2, h2 = list(map(float, bbox2.split()))

    if (abs(x2 - x1) <= 15 and abs(y2 - y1) <= 15) and (abs(w2 - w1) <= 15 and abs(abs(h2 - h1)) <= 15):
        return True
    return False


def denoise_image(img_name):
    image = cv2.imread('../pose_estimation/images/' + img_name)
    sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpen = cv2.filter2D(image, -1, sharpen_kernel)
    blur = cv2.bilateralFilter(sharpen, 7, 50, 50)
    cv2.imwrite('../pose_estimation/denoised_test/' + img_name, blur)
