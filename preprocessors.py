import numpy as np
import cv2
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin


class TargetEncoder(BaseEstimator,TransformerMixin
                    ):
    def__init__(self,encoder = LabelEncoder()):
    self.encoder = encoder

    def fit(self,x,y=None):
        # 'x' is the target in this case
        self.encoder.fit(x)
        return self
    def transform(self,x):
        x = x.copy()
        x = np_utils.to_categorical(self.encoder.transform(x))
        return x

    def _im_resize(df, n):
        im = cv2.imread(df[n])
        im = cv2.resize(im, (IMAGE_SIZE, IMAGE_SIZE))
        return im

    class CreateDataset(BaseEstimator,TransformerMixin):
        def __init__(self,image_size = 50):
            self.image_size = image_size

        def fit(self,x,y=None):
            return self

        def transform(self,x):
            x = x.copy()
            tmp = np.zeros((len(x),
                            self.image_size,
                            self.image_size, 3),
                           dtype='float32')

            for n in range(0, len(x)):
                im = _im_resize(x,n,self.image_size)
                tmp[n] = im


            print('Dataset Images shape:() size:(:,)'.format(tmp.shape,tmp.size))



