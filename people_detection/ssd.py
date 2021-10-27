import torch
import os
from people_detection.video_handler import Handler
import cv2
import re
import sys
import glob


def check_false_positive(actual_frame, x, y, w, h, original_width, original_height):
    labels = glob.glob("./inference/output/*.txt")
    labels.sort(key=natural_keys)
    actual = 0
    to_return = []

    substring = '_' + str(actual_frame) + '.txt'
    res = [i for i in labels if substring in i]

    if len(res) == 0:
        to_return.append(False)
        return to_return

    res = str(res[0].replace("\\", "/"))

    try:
        file = open(res, 'r')

        # all the bboxes of the label are considered(related to the paintings)
        while True:
            try:
                line = file.readline().split()
                new_w = int(float(line[3]) * original_width)
                new_h = int(float(line[4]) * original_height)
                new_x = int((float(line[1]) * original_width) - new_w / 2)
                new_y = int((float(line[2]) * original_height) - new_h / 2)
            except:
                break

            # if the new bbox of the person is contained in the one of the painting
            if (x >= new_x and y >= new_y) and (w <= new_w and h <= new_h):
                to_return.append(False)
            else:
                to_return.append(True)

        file.close()
    except:
        print('cannot open the file %s' % labels[actual])

    actual += 1
    return to_return


def atof(text):
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval


def natural_keys(text):
    return [atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text)]


def ssd(filename):
    max_people = 0
    folder = '../people_detection/images/'
    precision = 'fp32'
    handler = Handler()
    ssd_model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd', model_math=precision)

    utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')
    ssd_model.to('cuda')
    ssd_model.eval()

    try:
        os.mkdir('../people_detection/output')
    except OSError:
        print("Creation of the directory output failed")

    handler.handle(filename)

    images_from_folder = [folder + name for name in os.listdir(folder)]
    images_from_folder.sort(key=natural_keys)

    # the folder related to the video is created in the output destination folder
    try:
        os.mkdir('../people_detection/labels')
    except OSError:
        print("Creation of the directory people_labels failed in output test")

    try:
        os.mkdir('../people_detection/output_test')
    except OSError:
        print("Creation of the directory output_test failed in output test")
    else:
        print("Successfully created the directory in output test")

    try:
        os.mkdir('../people_detection/reshaped_images/')
    except OSError:
        print("Creation of the directory reshaped_images failed in reshaped images")
    else:
        print("Successfully created the directory in reshaped images")

    original_height, original_width, layers = cv2.imread(images_from_folder[0]).shape

    for uri in images_from_folder:
        to_resize = cv2.resize(cv2.imread(uri), (300, 300))
        cv2.imwrite('../people_detection/reshaped_images/' +
                    os.listdir(folder)[images_from_folder.index(uri)], to_resize)

    uris = ['../people_detection/reshaped_images/' + name for name in
            os.listdir('../people_detection/reshaped_images/')]

    inputs = [utils.prepare_input(uri) for uri in uris]
    index = 0

    while True:
        if (index + 1) * 50 < len(inputs):
            tensor = utils.prepare_tensor(inputs[index * 50:(index + 1) * 50], precision == 'fp16')
        else:
            tensor = utils.prepare_tensor(inputs[index * 50:-1], precision == 'fp16')

        with torch.no_grad():
            detections_batch = ssd_model(tensor)

        results_per_input = utils.decode_results(detections_batch)
        best_results_per_input = [utils.pick_best(results, 0.40) for results in results_per_input]

        # results are saved
        for image_idx in range(len(best_results_per_input)):
            # ...with detections
            image = cv2.imread(images_from_folder[image_idx + (index * 50)])
            bboxes, classes, confidences = best_results_per_input[image_idx]
            count = 0

            for idx in range(len(bboxes)):

                file = open('../people_detection/labels/frame%d.txt' % (image_idx + (index * 50)), 'a')
                if classes[idx] == 1:
                    left, bot, right, top = bboxes[idx]

                    new_w = int((right - left) * original_width)
                    new_h = int((top - bot) * original_height)
                    new_x = int(left * original_width)
                    new_y = int(bot * original_height)

                    file.write("%s %s %s %s \n" % (new_x, new_y, new_w, new_h))

                    in_the_picture = check_false_positive(image_idx + (index * 50), new_x, new_y, new_w, new_h,
                                                          original_width, original_height)

                    if all(in_the_picture):
                        cv2.rectangle(image, (new_x, new_y), (new_x + new_w, new_y + new_h), color=(0, 255, 0),
                                      thickness=3)
                        cv2.putText(image, 'Person ' + str(round(confidences[idx] * 100)) + '%',
                                    (new_x, new_y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)
                        count += 1

                    file.close()
            cv2.imwrite("../people_detection/output_test/frame" + str(image_idx + (index * 50)) + ".jpg", image)

            if count > max_people:
                max_people = count

        if index == int(len(inputs) / 50):
            break
        index += 1

    # the output video is created
    images = [img for img in os.listdir('../people_detection/output_test/')]
    images.sort(key=natural_keys)
    out = cv2.VideoWriter('../people_detection/output/result_video.avi', 0, 30, (original_width, original_height))

    for frame in images:
        out.write(cv2.imread(os.path.join('../people_detection/output_test/' + frame)))
    out.release()

    file = open('./people_found.txt', 'a+')
    file.write(str(max_people))
    file.close()


if __name__ == '__main__':
    ssd(sys.argv[1])
