import cv2
import os


def database_kp():
    sift = cv2.SIFT_create()
    kp = []
    des = []
    dir = '../sift/images/'

    for image in os.listdir(dir):
        actual_kp, actual_des = sift.detectAndCompute(cv2.imread(dir + image, cv2.IMREAD_GRAYSCALE), None)
        kp.append(actual_kp)
        des.append(actual_des)
    return kp, des


def distance_evaluation(distances):
    total_distance = 0
    i = 0

    for distance in distances:
        total_distance += distance.distance

        if distance.distance < 300:
            i += 1

    return total_distance, i  # the mean distance of the key points is returned


def localization(kp, im_h, im_w):
    # the image is divided in 4 boxes

    x = kp.pt[0]
    y = kp.pt[1]
    parts = 2

    if 0 <= x <= im_w / parts:
        pos_x = 0
    elif im_w / parts < x <= (2 * im_w) / parts:
        pos_x = 1

    if 0 <= y <= im_h / parts:
        pos_y = 0
    elif im_h / parts < y <= (2 * im_h) / parts:
        pos_y = 1

    return pos_x, pos_y


def comparison():
    path = '../rectification/painting_rect1/'
    already_saved = []
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    db_kps, db_des = database_kp()
    sift = cv2.SIFT_create()
    all_matches = {}

    all_paintings = os.listdir('../sift/images')

    for filename in os.listdir("../rectification/painting_rect1"):
        correspondences = {}
        picture = cv2.imread(path + filename, cv2.IMREAD_GRAYSCALE)

        # we suppose to be able to avoid at about all the external part of the painting without loosing interesting
        # details by cutting the 20% of the image
        x, y = picture.shape
        cut = picture[int(x / 5):int(x - (x / 5)), int(y / 5):int(y - (y / 5))]
        kp, des = sift.detectAndCompute(cut, None)

        for index in range(len(db_kps)):
            matches = bf.match(des, db_des[index])

            correspondences[all_paintings[index]] = list(distance_evaluation(matches))
            all_matches[os.listdir('../sift/images/')[index]] = matches

        copy = correspondences

        first = \
            list({k: v for k, v in sorted(correspondences.items(), key=lambda item: (item[1][0], item[1][1]))}.items())[
                0]
        key_first = list(copy.keys())[list(copy.values()).index(first[1])]
        best_match = all_matches[key_first]
        actual_db_image = cv2.imread('../sift/images/' + key_first)
        same_sector = 0

        for actual_match in best_match:
            actual_query = kp[actual_match.queryIdx]
            actual_train = db_kps[list(os.listdir('../sift/images/')).index(key_first)][
                actual_match.trainIdx]

            if localization(actual_query, cut.shape[0], cut.shape[1]) == localization(actual_train,
                                                                                      actual_db_image.shape[0],
                                                                                      actual_db_image.shape[1]):
                same_sector += 1

        if same_sector >= 0.55 * len(best_match) and first[0] not in already_saved:
            already_saved.append(first[0])

    file = open('./images_found.txt', 'a+')

    for element in already_saved:
        file.write(element + '\n')
    file.close()


if __name__ == '__main__':
    comparison()
