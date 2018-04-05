""" Test Predictor Class """
from predicting.predictor import Predictor

root_path = "D:\\Studium_GD\\Zooniverse\\Data\\camtrap_trainer\\data\\cats_and_dogs\\models\\cats_vs_dogs\\"
image_dir = 'D:\\Studium_GD\\Zooniverse\\CamCatProject\\data\\sample_images_raw\\'

pred = Predictor(
    model_path=root_path + "model_prediction_run_201804050704.hdf5",
    class_mapping_json=root_path + "label_mappings.json",
    pre_processing_json=root_path + "image_processing.json",
    batch_size=128)

predictions = pred.predict_image_dir(image_dir)
