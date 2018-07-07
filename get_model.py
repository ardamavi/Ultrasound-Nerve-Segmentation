# Arda Mavi
import os
import numpy as np
from keras.models import Model
from keras import backend as K
from keras.models import model_from_json
from keras.layers import Input, Conv2D, UpSampling2D, Activation, MaxPooling2D, Flatten, Dense, concatenate, Dropout

def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def save_model(model):
    if not os.path.exists('Data/Model/'):
        os.makedirs('Data/Model/')
    model_json = model.to_json()
    with open("Data/Model/model.json", "w") as model_file:
        model_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("Data/Model/weights.h5")
    print('Model and weights saved')
    return

def get_model():

    inputs = Input(shape=(256, 256, 1))
    
    conv_block_1 = Conv2D(16, (3, 3), strides=(1, 1), padding='same')(inputs)
    conv_block_1 = Activation('relu')(conv_block_1)
    conv_block_1 = Conv2D(16, (3, 3), strides=(1, 1), padding='same')(conv_block_1)
    conv_block_1 = Activation('relu')(conv_block_1)
    pool_block_1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv_block_1)
    
    conv_block_2 = Conv2D(32, (3, 3), strides=(1, 1), padding='same')(pool_block_1)
    conv_block_2 = Activation('relu')(conv_block_2)
    conv_block_2 = Conv2D(32, (3, 3), strides=(1, 1), padding='same')(conv_block_2)
    conv_block_2 = Activation('relu')(conv_block_2)
    pool_block_2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv_block_2)
    
    conv_block_3 = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(pool_block_2)
    conv_block_3 = Activation('relu')(conv_block_3)
    conv_block_3 = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(conv_block_3)
    conv_block_3 = Activation('relu')(conv_block_3)
    pool_block_3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv_block_3)
    
    conv_block_4 = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(pool_block_3)
    conv_block_4 = Activation('relu')(conv_block_4)
    conv_block_4 = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(conv_block_4)
    conv_block_4 = Activation('relu')(conv_block_4)
    pool_block_4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv_block_4)
    
    conv_block_5 = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(pool_block_4)
    conv_block_5 = Activation('relu')(conv_block_5)
    conv_block_5 = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(conv_block_5)
    conv_block_5 = Activation('relu')(conv_block_5)
    
    up_block_1 = UpSampling2D((2, 2))(conv_block_5)
    up_block_1 = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(up_block_1)
    
    merge_1 = concatenate([conv_block_4, up_block_1])
    
    conv_block_6 = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(merge_1)
    conv_block_6 = Activation('relu')(conv_block_6)
    conv_block_6 = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(conv_block_6)
    conv_block_6 = Activation('relu')(conv_block_6)
    
    up_block_2 = UpSampling2D((2, 2))(conv_block_6)
    up_block_2 = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(up_block_2)
    
    merge_2 = concatenate([conv_block_3, up_block_2])
    
    conv_block_7 = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(merge_2)
    conv_block_7 = Activation('relu')(conv_block_7)
    conv_block_7 = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(conv_block_7)
    conv_block_7 = Activation('relu')(conv_block_7)
    
    up_block_3 = UpSampling2D((2, 2))(conv_block_7)
    up_block_3 = Conv2D(32, (3, 3), strides=(1, 1), padding='same')(up_block_3)
    
    merge_3 = concatenate([conv_block_2, up_block_3])
    
    conv_block_8 = Conv2D(32, (3, 3), strides=(1, 1), padding='same')(merge_3)
    conv_block_8 = Activation('relu')(conv_block_8)
    conv_block_8 = Conv2D(32, (3, 3), strides=(1, 1), padding='same')(conv_block_8)
    conv_block_8 = Activation('relu')(conv_block_8)
    
    up_block_4 = UpSampling2D((2, 2))(conv_block_8)
    up_block_4 = Conv2D(16, (3, 3), strides=(1, 1), padding='same')(up_block_4)
    
    merge_4 = concatenate([conv_block_1, up_block_4])
    
    conv_block_9 = Conv2D(16, (3, 3), strides=(1, 1), padding='same')(merge_4)
    conv_block_9 = Activation('relu')(conv_block_9)
    conv_block_9 = Conv2D(16, (3, 3), strides=(1, 1), padding='same')(conv_block_9)
    conv_block_9 = Activation('relu')(conv_block_9)
    
    conv_block_10 = Conv2D(1, (1, 1), strides=(1, 1), padding='same')(conv_block_9)
    outputs = Activation('sigmoid')(conv_block_10)
    
    model = Model(inputs=inputs, outputs=outputs)


    model.compile(optimizer='adadelta', loss=dice_coef_loss, metrics=[dice_coef])

    return model

if __name__ == '__main__':
    save_model(get_model())
