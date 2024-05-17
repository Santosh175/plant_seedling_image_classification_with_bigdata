from sklearn.pipeline import Pipeline

import config
import preprocessors as pp
import model
pipe = Pipeline([
    ('dataset',pp.CreateDataset(config.IMAGE_SIZE)),
    ('cnn_model',model.cnn_clf)

])

if__name__=='__main__':

  from sklearn.metrics import accuracy_score
  import data_management as dm
  import config