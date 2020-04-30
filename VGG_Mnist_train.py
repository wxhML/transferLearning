






import time
# sys.setdefaultencoding('utf-8')
from keras.datasets import mnist
import gc
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Flatten
from keras.applications.vgg16 import VGG16
from keras.optimizers import SGD
import cv2
import numpy as np
from keras.applications import VGG16,resnet50


time1 = time.time()
def tran_y(y):
    y_ohe = np.zeros(10)
    y_ohe[y] = 1
    return y_ohe
ishape=48
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = [cv2.cvtColor(cv2.resize(i, (ishape, ishape)), cv2.COLOR_GRAY2BGR) for i in X_train]
X_train = np.concatenate([arr[np.newaxis] for arr in X_train]).astype('float32')
X_train /= 255.0

X_test = [cv2.cvtColor(cv2.resize(i, (ishape, ishape)), cv2.COLOR_GRAY2BGR) for i in X_test]
X_test = np.concatenate([arr[np.newaxis] for arr in X_test]).astype('float32')
X_test /= 255.0

y_train_ohe = np.array([tran_y(y_train[i]) for i in range(len(y_train))])
y_test_ohe = np.array([tran_y(y_test[i]) for i in range(len(y_test))])
y_train_ohe = y_train_ohe.astype('float32')
y_test_ohe = y_test_ohe.astype('float32')
print(X_train.shape)
for i in range(10):
    gc.collect()
ishape=48
model_vgg = VGG16(include_top = False, weights = 'imagenet', input_shape = (ishape, ishape, 3))
for layer in model_vgg.layers:
        layer.trainable = False
model = Flatten()(model_vgg.output)
model = Dense(4096, activation='relu', name='fc1')(model)
model = Dense(4096, activation='relu', name='fc2')(model)
model = Dropout(0.5)(model)
model = Dense(10, activation = 'softmax', name='prediction')(model)
model_vgg_mnist_pretrain = Model(model_vgg.input, model, name = 'vgg16_pretrain')
print(model_vgg_mnist_pretrain.summary())
sgd = SGD(lr = 0.05, decay = 1e-5)
model_vgg_mnist_pretrain.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])
model_vgg_mnist_pretrain.fit(X_train, y_train_ohe, validation_data = (X_test, y_test_ohe), epochs = 10, batch_size = 64)
model.save('model_vgg16_minist.h5')
scores=model_vgg_mnist_pretrain.evaluate(X_test,y_test_ohe,verbose=0)
print(scores)
time2 = time.time()
print(u'ok,结束!')
print(u'总共耗时：' + str(time2 - time1) + 's')
