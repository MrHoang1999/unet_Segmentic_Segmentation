from keras.callbacks import ModelCheckpoint

from model import *
from dataset import *

data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
image_size = 256
batch_size = 3
train_path ="D:\\segment\\dataset"
list_ids = get_listIDS(train_path)
GenTrain = DataSequence(list_ids, train_path, batch_size, image_size,data_gen_args)
train_steps = 10
model = UNet()
model_checkpoint = ModelCheckpoint('detect.h5', monitor='loss',verbose=1, save_best_only=True)
callbacks_list =[model_checkpoint]
model.fit_generator(GenTrain,steps_per_epoch=5, epochs=5,callbacks=callbacks_list)
model.save('detect.h5')