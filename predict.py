import data_management as dm
import config


def make_prediction(*,path_to_images) -> float:
    """Make a prediction using the saved model pipeline"""

    # Load data
    # create a dataframe with columns = ['image','target']
    # column "image" contains path to image
    # columns target can contain all zeros, it doesn't matter

    dataframe = path_to_images # needs to load as above described
    pipe = dm.load_pipeline_keras()
    predictions = pipe.pippe.predict(dataframe)

    # response = {'predictions': predictions, 'versions':_version}

    return predictions

    return predictions