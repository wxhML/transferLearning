from keras.models import load_model
from keras.preprocessing.image import img_to_array,load_img
import json
import numpy as np
import matplotlib.pyplot as plt

file = open('label.json','r',encoding='utf-8')
label = json.load(file)

# 载入模型
model = load_model('model_vgg16_dog.h5')
def predict(image):
    # 导入图片
    image = load_img(image)
    plt.imshow(image)
    image = image.resize((150,150))
    image = img_to_array(image)
    image = image/255
    image = np.expand_dims(image,0)   
    plt.title(label[str(model.predict_classes(image)[0])])
    plt.axis('off')
    plt.show() 
predict('data/test/n02094433-Yorkshireterrier/YorkshireTerrier-135775o2.jpg')