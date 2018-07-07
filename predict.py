# Arda Mavi
import sys
import numpy as np
from get_dataset import get_img, save_img
from keras.models import model_from_json

def predict(model, X):
    X = X.reshape(1, 256, 256, 1)
    Y = model.predict(X)*255.
    return Y

if __name__ == '__main__':
    img_dir = sys.argv[1]
    img = get_img(img_dir)/255.
    # Getting model:
    model_file = open('Data/Model/model.json', 'r')
    model = model_file.read()
    model_file.close()
    model = model_from_json(model)
    # Getting weights
    model.load_weights("Data/Model/weights.h5")
    Y = predict(model, img)
    name = 'mask.png'
    save_img(Y, name)
    print('Segmentated image saved as '+name)
