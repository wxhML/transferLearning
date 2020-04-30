# transferLearning
识别项目的入门级
VGG网络的汪星人识别，手写字体识别
主要通过汪星人的数据集和手写数据集进行训练，然后进行预测

1：下载数据集
汪星人的数据链接：https://pan.baidu.com/s/1meTMknHTCkVmD-k3sZsF9Q 密码：9hz6

2：配置环境
pip install -r requriements.txt
也可以自行安装：tensorflow>=1.13即可。

3：运行VGG_Dog_predict.py，进行识别汪星人网络的训练；
   运行VGG_Dog_train.py，加载训练的模型，进行汪星人的识别
   本模型训练的准确率只有80%多，原因
   
   1：修改主干网络：汪星人的识别属于细粒度分类，可以采用其他网络进行分类，例如resnet50，inception等；
   
   2：增加汪星人的数据集；
   
   运行VGG_Mnist_train.py，进行MINST数据的识别，识别率可以达到97%,增加训练的次数，可以达到99%以上。


