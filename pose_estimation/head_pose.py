import argparse
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from hopenet import Hopenet
from PIL import Image
from torchvision import transforms
from visualization import draw_pose
import os
import re
import math
from generic_checks import check, denoise_image
from eye_detector import face_detection, eyes_detection


def atof(text):
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval


def natural_keys(text):
    return [atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text)]


def compare_with_paintings(x, y, x_angle, frame_name, w, h):
    to_return = []

    try:
        actual_label = open('../interface/inference/output/' + frame_name)
    except:
        to_return.append(False)
        return to_return

    all_paintings = actual_label.readlines()

    for painting in all_paintings:
        painting = painting.split()[1:]
        painting = list(map(float, painting))

        # we consider the center of the bounding box, so the center of the painting and the center of the face of the
        # person and then we compute the angle between them and if the person's face inclination degree is contained
        # between the estimation person-bottom of the painting and person-top of the painting, the person is looking it

        painting_y_high = (painting[1] - (painting[-1] / 2)) * h
        painting_y_low = (painting[1] + (painting[-1] / 2)) * h
        painting_x_high = (painting[0] - (painting[-1] / 2)) * w
        painting_x_low = (painting[0] + (painting[-1] / 2)) * w

        person_superior = math.degrees(math.atan2(x - painting_x_high, y - painting_y_high))
        person_inferior = math.degrees(math.atan2(x - painting_x_low, y - painting_y_low))

        if person_inferior <= x_angle <= person_superior:
            to_return.append(True)
        else:
            to_return.append(False)
    return to_return


class HeadPose:
    def __init__(self, checkpoint, transform=None):
        self.transform = transform
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        num_bins = 66
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.idx_tensor = torch.FloatTensor([idx for idx in range(num_bins)]).to(self.device)
        self.model = Hopenet()
        checkpoint = torch.load(checkpoint)
        self.model.load_state_dict(checkpoint, strict=False)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, image):
        if isinstance(image, torch.Tensor) and len(image.shape) == 4:
            data = torch.stack([self.transform(transforms.functional.to_pil_image(img)) for img in image])
        if isinstance(image, torch.Tensor):
            data = self.transform(transforms.functional.to_pil_image(image))
        elif isinstance(image, str):
            image = Image.open(image)
            data = self.transform(image).unsqueeze(dim=0)
        else:
            data = self.transform(image).unsqueeze(dim=0)

        data = data.to(self.device)
        yaw, pitch, roll = self.model(data)
        yaw = F.softmax(yaw, dim=1)
        pitch = F.softmax(pitch, dim=1)
        roll = F.softmax(roll, dim=1)

        yaw = torch.sum(yaw * self.idx_tensor, dim=1) * 3 - 99
        pitch = torch.sum(pitch * self.idx_tensor, dim=1) * 3 - 99
        roll = torch.sum(roll * self.idx_tensor, dim=1) * 3 - 99
        return {'yaw': yaw, 'pitch': pitch, 'roll': roll}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='../pose_estimation/data/hopenet.pkl', type=str)
    parser.add_argument('--image', type=str)
    args = parser.parse_args()
    mov = False

    try:
        os.mkdir("../pose_estimation/images")
    except:
        print('folder images in head pose still exists')

    try:
        os.mkdir('../pose_estimation/denoised_test')
    except:
        print('folder denoised_test in head pose still exists')

    frames = os.listdir('../people_detection/images')

    labels = os.listdir('../people_detection/labels')
    labels.sort(key=natural_keys)
    frames.sort(key=natural_keys)

    count = 0
    all_images = os.listdir('../people_detection/images')
    all_images.sort(key=natural_keys)
    name = os.listdir('./inference/output/')[1]
    name = name[:name.rfind('_')]

    original_w, original_h, _ = cv2.imread('../people_detection/images/' +
                                           os.listdir('../people_detection/images/')[1]).shape

    for file in labels:
        label = open('../people_detection/labels/' + file, 'r')
        old_line = []

        present = False
        for actual_frame in all_images:
            if actual_frame[:-4] == file[:-4]:
                index = all_images.index(actual_frame)
                present = True
                break

        if not present:
            continue

        actual = 0

        for line in label.readlines():
            x, y, w, h = line.split()

            if line == old_line:
                continue

            x = int(x)
            y = int(y)
            w = int(w)
            h = int(h)

            if x < 0:
                x = 0

            if y < 0:
                y = 0

            if w < 0:
                w = 0

            if h < 0:
                h = 0

            frame_number = int(all_images[index][all_images[index].rfind('e') + 1:-4])
            image = cv2.imread('../people_detection/images/' + all_images[index])
            cv2.imwrite("../pose_estimation/images/" + name + "_%d" % frame_number + "_%d.jpg" % actual,
                        image[y:y + h, x: x + w])
            actual += 1
            old_line = line
    label.close()

    old_name = ''
    old_coordinates = []
    returned = -1

    for image in os.listdir('../pose_estimation/images'):
        denoise_image(image)

    all_images = os.listdir('../pose_estimation/denoised_test')
    all_images.sort(key=natural_keys)

    for image in all_images:
        looking = True
        eyes = 0
        actual_image = '../pose_estimation/denoised_test/' + image

        index = str(image[image.rfind('_', 0, image.rfind('_')) + 1:image.rfind('_')])

        if str('frame' + index + '.txt') not in labels:
            continue

        actual_label = labels[labels.index('frame' + index + '.txt')]
        bbox = open('../people_detection/labels/' + actual_label)

        all_bbox = bbox.readlines()

        for bbox in all_bbox:
            bbox = bbox.split()

            # The same person is in motion so he cannot be facing a painting
            if (old_name == actual_image and old_coordinates == bbox) or returned >= 0:
                returned = -1
                continue

            head_pose = HeadPose(checkpoint=args.checkpoint)
            angles = head_pose.predict(actual_image)
            yaw, pitch, roll = angles['yaw'].item(), angles['pitch'].item(), angles['roll'].item()

            img = cv2.imread(actual_image)
            img, angle_x, angle_y, angle_z = draw_pose(img, yaw, pitch, roll, tdx=100, tdy=200, size=100)

            old_name = actual_image
            face_x, face_y, face_w, face_h = face_detection(actual_image)

            # no face was detected but the person is present: he's likely looking at a painting
            if face_x == -1 and face_y == -1 and face_w == -1 and face_h == -1:
                looking = False

                # returns the number of eyes detected: if it's 2 the person is looking the camera, so he's not
                # looking to a painting
                eyes = eyes_detection(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), actual_image)
            else:
                continue

            # we assume that: the center of the face(considering the bbox of the person) can reveal the position of
            # the eyes the assumption is that: the center of the face is at about at w/2, 75 because of some tests
            if any(compare_with_paintings(float(bbox[0]) + (float(bbox[2]) / 2), float(bbox[1]) + 75, angle_x,
                                          image[:image.rfind('_')] + '.txt', original_w, original_h)) and \
                    not looking and \
                    eyes < 2:

                if all_images.index(image) > 0:
                    print('here a person is looking at a painting', image)
                    present_image = cv2.imread('../pose_estimation/images/' + image)
                    cv2.imwrite('../pose_estimation/test_result/' + image, present_image)
