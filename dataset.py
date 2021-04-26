import cv2
import os
from tensorflow import keras
import numpy as np
from labelling import *
background=[0,0,0]
Screen=[54,209,10]
Speaker=[107,133,136]
aircondition=[159,125,31]
board=[43,110,205]
ceilingfan=[160,64,64]
chair=[96,64,64]
fan=[124,141,104]
poweroutlet=[129,130,71]
projectors=[77,180,86]
table=[176,128,192]
tubelight=[124,253,199]
window=[176,32,128]
COLOR_DICT = []
COLOR_DICT.append(background)
COLOR_DICT.append(Screen)
COLOR_DICT.append(Speaker)
COLOR_DICT.append(aircondition)
COLOR_DICT.append(board)
COLOR_DICT.append(ceilingfan)
COLOR_DICT.append(chair)
COLOR_DICT.append(fan)
COLOR_DICT.append(poweroutlet)
COLOR_DICT.append(projectors)
COLOR_DICT.append(table)
COLOR_DICT.append(tubelight)
COLOR_DICT.append(window)
def labelVisualize(num_class,color_dict,img):
    img = img[:,:,0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i,:] = color_dict[i]
    return img_out / 255.0

class DataSequence(keras.utils.Sequence):
    def __init__ (self, list_IDs, folder_image ,batch_size, image_size, n_channels=3, shuffle=True):
        self.list_IDs = list_IDs
        self.folder_image = folder_image
        self.batch_size = batch_size
        self.image_size = image_size
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __load__(self, ids_name):
        image_path = os.path.join(self.folder_image, "image", ids_name)
        mask_path = os.path.join(self.folder_image, "label", ids_name)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (self.image_size,self.image_size))

        mask = cv2.imread(mask_path)
        mask = cv2.resize(mask, (self.image_size, self.image_size))
        mask = cv2.cvtColor(mask,cv2.COLOR_BGR2RGB)
        mask = one_hot_it(mask,COLOR_DICT)
        return (image, mask)

    def __getitem__(self, index):

        if (index + 1) * self.batch_size > len(self.list_IDs):
            batch_file = self.list_IDs[index * self.batch_size:len(self.list_IDs)]
        else:
            batch_file = self.list_IDs[index * self.batch_size: (index + 1) * self.batch_size]

        image = []
        mask = []

        for name_file in batch_file:
            (_img, _mask) = self.__load__(name_file)
            image.append(_img)
            mask.append(_mask)

        image = np.asarray(image)
        mask = np.asarray(mask)

        return image, mask

    def on_epoch_end(self):
        pass

    def __len__(self):
        return int(np.ceil(len(self.list_IDs) / float(self.batch_size)))


def get_listIDS(path):
    path_image = os.path.join(path, "image")
    list_ids = os.listdir(path_image)

    print(len(list_ids))
    list_ids_cut = list_ids[:len(list_ids)]

    return list_ids_cut
