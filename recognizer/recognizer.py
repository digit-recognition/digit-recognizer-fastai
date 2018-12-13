import sys
sys.path.append('../')

from fastai.imports import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *
import traceback
from resizeimage import resizeimage


class Recognizer:
    def recognize(self, PATH, IMG_SIZE):
        img = Image.open(PATH / "data/gray2.png")
        img = resizeimage.resize('thumbnail', img, [28, 28])

        arr = np.array(img)
        arr = arr.reshape(1, 784)
        print(arr.shape)

        arr = np.repeat(arr, 3, axis=1)
        print(arr.shape)

        to_recognize = arr.reshape(1, 28, 28, 3)
        print(to_recognize.shape)

        arch = resnet34
        stats = (np.array([0.4914, 0.48216, 0.44653]), np.array([0.24703, 0.24349, 0.26159]))
        tfms = tfms_from_stats(stats, IMG_SIZE, aug_tfms=[RandomFlip()], pad=IMG_SIZE // 8)

        x_template = np.zeros((1, 28, 28, 3))
        y_template = np.zeros((1))

        data = ImageClassifierData.from_arrays(PATH, trn=(x_template, y_template), val=(x_template, y_template),
                                               test=to_recognize,
                                               tfms=tfms)

        data.trn_ds.c = 10  # num of classes

        learn = ConvLearner.pretrained(arch, data, precompute=False)
        learn.load(PATH / '28_all')
        print('loaded')

        log_preds, y = learn.TTA(is_test=True)  # use test dataset rather than validation dataset

        probs = np.mean(np.exp(log_preds), 0)
        actual = probs[0].argmax()

        print('correct =', 7, ' actual =', actual)

    def _recognize(self, PATH, IMG_SIZE):
        all_train = pd.read_csv(filepath_or_buffer=PATH / 'data/train.csv')
        print(all_train.shape)

        val_ids = get_cv_idxs(all_train['label'].size)
        all_train.drop(all_train.index[val_ids])
        valid = all_train.iloc[val_ids]
        print(valid.shape)

        n_channels = 3

        valid_y = np.array(valid['label'])

        tmp_x = np.array(valid.iloc[:, 1:])
        print(tmp_x.shape)

        valid_x = np.repeat(tmp_x, n_channels, axis=1)
        valid_x = valid_x.astype(np.float32)

        print(valid_x.shape)

        valid_x = valid_x.reshape(valid_x.shape[0], IMG_SIZE, IMG_SIZE, n_channels)
        print(valid_x.shape)

        arch = resnet34
        stats = (np.array([0.4914, 0.48216, 0.44653]), np.array([0.24703, 0.24349, 0.26159]))
        tfms = tfms_from_stats(stats, IMG_SIZE, aug_tfms=[RandomFlip()], pad=IMG_SIZE // 8)

        x_template = np.zeros((1, 28, 28, 3))
        y_template = np.zeros((1))
        to_recognize = valid_x[0].reshape(1, IMG_SIZE, IMG_SIZE, 3)

        data = ImageClassifierData.from_arrays(PATH, trn=(x_template, y_template), val=(x_template, y_template),
                                               test=to_recognize,
                                               tfms=tfms)
        data.trn_ds.c = 10  # num of classes

        learn = ConvLearner.pretrained(arch, data, precompute=False)
        learn.load(PATH / '28_all')
        print('loaded')

        log_preds, y = learn.TTA(is_test=True)  # use test dataset rather than validation dataset

        probs = np.mean(np.exp(log_preds), 0)
        actual = probs[0].argmax()

        print('correct =', valid_y[0], ' actual =', actual)


