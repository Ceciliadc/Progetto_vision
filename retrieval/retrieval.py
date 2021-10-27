import os
import pandas as pd
from scipy.io import wavfile


def data_retrieval(image_name):
    # all the data related to the found element are returned
    # they can be accessed by using the column name as done for data

    data = pd.read_csv("../sift/description/data.csv", encoding='ISO-8859-1')
    return data[data['Image'] == image_name]


def data_of_aspecific_image(title):
    data = pd.read_csv("../sift/description/data.csv", encoding='ISO-8859-1')
    info = data[data['Image'] == title]

    info = info.values.tolist()

    name = info[0][0]
    author = info[0][1]
    room = info[0][2]
    image = info[0][3]
    audio = info[0][7]

    try:
        description_file = open('../sift/descriptions/' + os.path.splitext(title)[0] + '.txt', 'r')
        description_file.close()
    except:
        print()

    return name, author, room, image, audio