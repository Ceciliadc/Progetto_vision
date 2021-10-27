import cv2


def face_detection(image_name):
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    image = cv2.imread(image_name, 0)
    image = image[:int(image.shape[0] / 3), :]
    faces = list(faceCascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=2))

    if len(faces) > 0:
        x = faces[0][0]
        y = faces[0][1]
        w = faces[0][2]
        h = faces[0][3]
    else:
        x = -1
        y = -1
        w = -1
        h = -1
    return x, y, w, h


def eyes_detection(x, y, w, h, image_name):
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    image = cv2.imread(image_name, 0)

    if x < 0 and y < 0 and w < 0 and h < 0:
        return 0

    eyes = eye_cascade.detectMultiScale(image[x: x + w, y:y + h], minNeighbors=2)
    number_eyes = len(eyes)

    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(image[x: x + w, y:y + h], (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 5)

    cv2.destroyAllWindows()

    return number_eyes
