import sys
from io import BytesIO

sys.path.append('../')

from fastai.conv_learner import *
from fastai.dataset import *
from fastai.plots import *
from resizeimage import resizeimage
import base64


class Recognizer:

    def __init__(self):
        self.img_size = 28
        self.work_path = Path('.')

    def recognize_by_str(self, str_img):
        print(str_img)

        imgdata = base64.b64decode(str_img)

        img = Image.open(BytesIO(imgdata))

        print()
        print(img)

        return self.__recognize(img)

    def recognize_by_path(self, file_path):
        print(file_path)

        img = Image.open(file_path)
        return self.__recognize(img)

    def __recognize(self, img):
        img = resizeimage.resize('thumbnail', img, [28, 28])
        arr = np.array(img)
        print(arr)

        arr = arr.reshape(1, 784)
        print(arr.shape)

        arr = np.repeat(arr, 3, axis=1)
        print(arr.shape)

        to_recognize = arr.reshape(1, 28, 28, 3)
        print(to_recognize.shape)

        arch = resnet34
        stats = (np.array([0.4914, 0.48216, 0.44653]), np.array([0.24703, 0.24349, 0.26159]))
        tfms = tfms_from_stats(stats, self.img_size, aug_tfms=[RandomFlip()], pad=self.img_size // 8)

        x_template = np.zeros((1, 28, 28, 3))
        y_template = np.zeros((1))

        data = ImageClassifierData.from_arrays(self.work_path, trn=(x_template, y_template), val=(x_template, y_template),
                                               test=to_recognize,
                                               tfms=tfms)

        data.trn_ds.c = 10  # num of classes

        learn = ConvLearner.pretrained(arch, data, precompute=False)
        learn.load(self.work_path / '28_all')
        print('loaded')

        log_preds, y = learn.TTA(is_test=True)  # use test dataset rather than validation dataset

        probs = np.mean(np.exp(log_preds), 0)
        actual = probs[0].argmax()

        print('recognition result =', actual)

        return actual
