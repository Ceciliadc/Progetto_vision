import cv2
import os

input_dir = '../rectification/painting_rect/'
images = os.listdir(input_dir)
folder_images = os.listdir(input_dir)

# skips all the images that are formed by a single color
for elem in folder_images:
    image = cv2.imread(input_dir + elem)
    for part in image:
        if len(image[image == part]) >= 0.9 * (image.shape[0] * image.shape[1] * image.shape[2]):
            os.remove(input_dir + elem)
            break
