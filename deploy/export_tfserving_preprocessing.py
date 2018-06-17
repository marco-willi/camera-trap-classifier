""" Export a trained model to Tensorflow Serving with pre-processing included """
import tensorflow as tf
from pre_processing.image_transformations import preprocess_image
from data_processing.utils import  read_json
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import (
        load_model, model_from_json)
from tensorflow.contrib.session_bundle import exporter
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import (
    tag_constants, signature_constants, signature_def_utils_impl)

# Parameters
model_path = './test_big/cats_vs_dogs_multi/model_save_dir/prediction_model.hdf5'
label_mappings = './test_big/cats_vs_dogs_multi/model_save_dir/label_mapping.json'
pre_processing = './test_big/cats_vs_dogs_multi/model_save_dir/image_processing.json'
deploy_path = './test_big/cats_vs_dogs_multi/deploy_preproc/'
deploy_version = 1

# Remote parameters
# git clone -b deploy_models https://github.com/marco-willi/camera-trap-classifier.git ~/code/camera-trap-classifier
model_path = '/host/data_hdd/ctc/ss/example/saves/prediction_model.hdf5'
label_mappings = '/host/data_hdd/ctc/ss/example/saves/label_mapping.json'
pre_processing = '/host/data_hdd/ctc/ss/example/saves/image_processing.json'
deploy_path = '/host/data_hdd/ctc/ss/example/deploy_preproc/'
deploy_version = 1



model = load_model(model_path)
model_json = model.to_json()
model_weights = model.get_weights()
pre_processing = read_json(pre_processing)


sess = tf.Session()
K.set_session(sess)
K._LEARNING_PHASE = tf.constant(0)
K.set_learning_phase(0)


new_model = model_from_json(model_json)
new_model.set_weights(model_weights)


x_input = tf.placeholder(tf.float32, name='image', shape=(None, None, 3))
x_processed = preprocess_image(x_input, **pre_processing)
x_processed = tf.expand_dims(x_processed, 0)
x_processed = Input(tensor=x_processed, name='image')


new_model = model_from_json(model_json)
new_model.set_weights(model_weights)
new_model.layers.pop(0)
new_model.summary()
new_outputs = new_model(x_processed)
new_model2 = Model(x_processed, new_outputs)
new_model2.summary()

prediction_signature = tf.saved_model.signature_def_utils.predict_signature_def(
        {"image": new_model.input},
        {"label/species": new_model.output[0],
         "label/standing": new_model.output[1]})

# export_path is a directory in which the model will be created
builder = saved_model_builder.SavedModelBuilder(deploy_path)
legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')


# Initialize global variables and the model
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init_op)
builder.add_meta_graph_and_variables(
      sess, [tag_constants.SERVING],
      signature_def_map={
           signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
           prediction_signature,
      },
      legacy_init_op=legacy_init_op)

# save the graph
builder.save()
