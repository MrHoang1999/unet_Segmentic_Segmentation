from model import *
import cv2
import numpy as np
import matplotlib.pyplot as plt
from labelling import *
model = UNet()
model.load_weights("detect.h5")
image_size = 256
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
img =  cv2.imread("D:\\segment\\dataset\\image\\3960.jpg")
#img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img = cv2.resize(img, (256,256))
img=img
X =  np.expand_dims(img, axis =0)
pred =  model.predict(X)

pred = colour_code_segmentation(reverse_one_hot(pred[0]),COLOR_DICT)
img = cv2.imread("D:\\segment\\dataset\\label\\3960.jpg")
fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
ax.imshow(img)
ax = fig.add_subplot(1, 2, 2)
ax.imshow(pred)
plt.show()