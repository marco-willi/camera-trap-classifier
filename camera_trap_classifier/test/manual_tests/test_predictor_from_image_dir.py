""" Test Predictor from Image Dir """
from predicting.predictor import Predictor

model_path = './test_big/cats_vs_dogs_multi/model_save_dir/best_model.hdf5'
pre_processing_json = './test_big/cats_vs_dogs_multi/model_save_dir/image_processing.json'
class_mapping_json = './test_big/cats_vs_dogs_multi/model_save_dir/label_mappings.json'
image_dir = './test/test_images/'
pred_output_csv = './test_big/cats_vs_dogs_multi/model_save_dir/preds.csv'
pred_output_json = './test_big/cats_vs_dogs_multi/model_save_dir/preds.json'
batch_size = 5

pred = Predictor(
            model_path=model_path,
            class_mapping_json=class_mapping_json,
            pre_processing_json=pre_processing_json,
            batch_size=batch_size)

pred.predict_from_image_dir(
    image_dir=image_dir,
    export_type='csv',
    output_file=pred_output_csv)

#pred.predict_from_image_dir(
#    image_dir=image_dir,
#    export_type='json',
#    output_file=pred_output_json)
