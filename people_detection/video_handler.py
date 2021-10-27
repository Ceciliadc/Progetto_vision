import cv2
import os


# division of the video in the frames it's composed of; it takes all the frames of the original video
class Handler:
    def handle(self, filepath):

        video_cap = cv2.VideoCapture(filepath)
        success, image = video_cap.read()
        count = 0
        new_path = '../people_detection/images/'
        new_path = os.path.splitext(new_path)[0]

        print('new path is', new_path)

        try:
            os.mkdir(new_path)
        except OSError:
            print("Creation of the directory %s failed" % os.path.splitext(new_path)[0])
        else:
            print("Successfully created the directory %s " % os.path.splitext(new_path)[0])

        while success:
            cv2.imwrite(new_path + '/frame%d.jpg' % count, image)
            success, image = video_cap.read()
            count += 1

