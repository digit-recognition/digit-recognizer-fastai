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

class Recognizer:

    def recognize(self, PATH, IMG_SIZE):

        test = pd.read_csv(filepath_or_buffer=PATH / 'data/test.csv')
        print(test.shape)

        all_train = pd.read_csv(filepath_or_buffer=PATH / 'data/train.csv')
        print(all_train.shape)

        val_ids = get_cv_idxs(all_train['label'].size)
        train = all_train.drop(all_train.index[val_ids])
        valid = all_train.iloc[val_ids]
        print(train.shape, valid.shape)

        n_channels = 3

        train_y = np.array(train['label'])
        train_x = np.repeat(np.array(train.iloc[:, 1:]), n_channels, axis=1)
        train_x = train_x.astype(np.float32)

        valid_y = np.array(valid['label'])
        valid_x = np.repeat(np.array(valid.iloc[:, 1:]), n_channels, axis=1)
        valid_x = valid_x.astype(np.float32)

        print(train_x.shape, valid_x.shape)

        train_x = train_x.reshape(train_x.shape[0], IMG_SIZE, IMG_SIZE, n_channels)
        valid_x = valid_x.reshape(valid_x.shape[0], IMG_SIZE, IMG_SIZE, n_channels)

        test_x = np.repeat(np.array(test.iloc[:, 0:]), n_channels, axis=1)
        test_x = test_x.astype(np.float32)

        test_x = test_x.reshape(test_x.shape[0], IMG_SIZE, IMG_SIZE, n_channels)
        print(test_x.shape)

        arch = resnet34
        stats = (np.array([0.4914, 0.48216, 0.44653]), np.array([0.24703, 0.24349, 0.26159]))
        tfms = tfms_from_stats(stats, IMG_SIZE, aug_tfms=[RandomFlip()], pad=IMG_SIZE // 8)
        data = ImageClassifierData.from_arrays(PATH, trn=(train_x, train_y), val=(valid_x, valid_y), test=test_x
                                               , tfms=tfms
                                               )

        learn = ConvLearner.pretrained(arch, data, precompute=False)
        learn.load(PATH / '28_all')
        print('loaded')

        to_recognize = valid_x[0].reshape(1, IMG_SIZE, IMG_SIZE, 3)

        data = ImageClassifierData.from_arrays(PATH, trn=(train_x, train_y), val=(valid_x, valid_y), test=to_recognize,
                                               tfms=tfms)
        learn.set_data(data)
        log_preds, y = learn.TTA(is_test=True)  # use test dataset rather than validation dataset

        probs = np.mean(np.exp(log_preds), 0)
        actual = probs[0].argmax()

        print('correct =', valid_y[0], ' actual =', actual)


