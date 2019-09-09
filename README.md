# 关于Face Normalization Model的使用（2019.08.08）

FNM模型基本使用情况的说明。</br>

需要配合fnm模型([https://github.com/mx54039q/fnm](https://github.com/mx54039q/fnm "fnm"))，以及MTCNN人脸检测模型使用。数据集为CASIA-WebFace和Multi-PIE。</br>论文作者使用的是Multi-PIE Session 1中Multiview的图片。在每个人的文件夹中，又分为面无表情和微笑表情的两组照片，文中只使用了面无表情的照片。另外在使用Multi-PIE的侧脸照片作为训练集时，15个不同角度中倒立的两个角度也被除外了。

---

## 1. 数据准备

根据论文中所说明的，fnm模型需要我们准备好三组数据：

- profile：也就是训练所需要的全部侧脸数据，这里使用的是CASIA-WebFace的全部图片。（需注意，根据论文中所说，他们使用了WebFace中的297369张照片，明显小于全数据集的大小的494414张照片。选取标准尚不明确。）

- front：也就是训练所需要的全部正脸数据，根据论文中所说，这里是用的是Multi-PIE Session 1的Multiview中，前150人的正脸图片，也即05_1号机位的图片，图片名格式大致为`xxx_01_01_051_xx.png`。

- test：也就是将要作为测试数据的侧脸照片，这里是用的是Multi-PIE Session 1的Multiview中，剩余100人的侧脸照片，也就是除了05_1和两个倒立角度外，剩余12个角度的照片。这些照片将会在模型跑完每一个epoch之后，自动生成对应的正脸照片，存在`fnm/test/fnm/epoch[0-9]`下。

fnm论文中需要我们把输入的图片全部裁剪到只剩下面部，并且大小设定为250×250像素，具体裁剪标准可以参考论文中的图片。这里使用的工具是MTCNN的面部检测功能。在`mtcnn/`目录下有我写好的两个文件，`demo.py`和`crop_image.py`。</br>

`demo.py`的主要作用是debug，确保人脸框在正确位置。使用时需要填入输入照片的文件夹位置和包含文件名的txt文件位置，然后`demo.py`会检测照片人脸位置，将五官和人脸框，以及以原人脸框为中心，向外扩张到250×250像素的人脸框画在照片上，在`mtcnn/`文件夹中生成名为`demo_`+文件名的新图片。</br>

`crop_image.py`除了需要输入照片的文件夹位置和图片名txt外，还需要一个存储位置文件夹名，运行后会在存储位置生成250×250像素的名为`c_`+文件名的新图片。</br>

注意：在输入文件到fnm时，记得根据实际情况，将`config.py`里面的`ori_height`和`ori_width`改成250。

这种在原来人脸框外额外选中250×250像素边框的办法，基本只适用于Multi-PIE里的图片。因为本身并没有做缩放，resize的工作，其他图片有可能因为像素问题，导致250×250框里人脸太大或者太小。还要想新的办法。

---

## 2. 数据导入

数据导入方面，我们需要把数据集所在位置信息，填写在`fnm/config.py`文件中。将照片所在的文件夹位置，填入config中的`'profile_path'`, `'front_path'`, `'test_path'`；将照片分别的文件名，以`.txt`文件的形式保存好，然后将txt文件位置存入config中的`'profile_list'`, `'front_list'`, `'test_list'`即可。</br>

例如：</br>
`flags.DEFINE_string('profile_path', '../Data/c_Multi-PIE/profile_list/', 'dataset path')`
`flags.DEFINE_string('profile_list', '../Data/c_Multi-PIE/profile_list/profile_list.txt', 'train profile list')`</br>
经测试`profile_list.txt`中的内容可以是纯文件名，如：`c_121_01_01_090_12.png`；也可以包含一部分的子文件夹名，如:`121/01/09_0/121_01_01_090_12.png`。

运行fnm只需要在python 2环境下，运行`python main.py`即可。在模型训练好后，想要预测一些图片的话，可以运行`python test.py --test_path ../../dataset/saveRejectFace --test_list ../fnm/mpie/school_profile.txt`，也是分别填入文件夹，和包含图片名的txt即可。注意这里`test.py`被改为了只预测txt中的前50张图片，可以根据需求去改。

注意：如果在此时，txt中的文件名包含子文件夹的话，则需要对`fnm/test.py`的保存图片路径进行微调，因为默认图片保存位置是`fnm/test/xxx.png`，如果路径中有进一步的子文件夹名就会出问题。

---

## 3. 监督学习(尚未完工)

现阶段的工作就是把fnm的模型改成监督学习，希望能够提高生成图片和原图的相似度。初步的设想其实很简单，就是在原有的`profile_list`和`front_list`以外，加一个`target_list`，作为每个侧脸照片的真实数据，然后算出侧脸照片和对应真实值之间的`feature_loss`，加入到模型中。</br>

由于还在初步构建阶段，目前`profile_list`选用的是Multi-PIE中的非正脸的图片，正脸图片则是根据profile中图片选择的对应的`xxx_01_01_051_10.png`图片。这样我们甚至不用在`config.py`里额外提供文件，只要在读取到`profile_list`之后，生成出对应的`target_list`就行了。将来的话可以考虑测算一下WebFace里人脸面向的角度，从而找出那个数据集里每个人的正脸照，作为目标图片。</br>

另外，由于不确定原本代码中的`tf.train.string_input_producer`能否确保profile和对应的target保持一致，我换用了`tf.FIFOQueue`工具，它可以同时将两个list做成queue，作用应该也是一样的。然后将target图片读取出来，resize到224×224像素，一起放入`shuffle_batch`应该就可以了。这些改动保存在`fnm/utils_1.py`中。</br>

随后在`WGAN_GP.py`的build up中，把target的feature通过face\_model做成feature，然后在loss中算出feature\_distance，加入到总的loss\_function中，应该也就可以了。这些改动保存在`fnm/WGAN_GP_1.py`中。</br>

在这之后，需要在`main.py`中做一点调整，包括使`get_train`额外输出一个target，使`build_up`额外多一个target作为输入，以及最后打印的loss中多出一个我们定义的新loss。这个loss在编写过程中我叫做supervised\_loss，但打印时考虑到这个词太长了，改名叫target\_loss了。</br>

目前的情况是，这些已经写出来了，但是新的fnm还跑不起来，到初始化完GPU之后就卡住了，既不往下进行，也不会报错，还需要下一步工作来处理。