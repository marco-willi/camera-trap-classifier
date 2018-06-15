""" Export a trained model to Tensorflow Serving """
import tensorflow as tf
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
deploy_path = './test_big/cats_vs_dogs_multi/deploy/'
deploy_version = 1

model_path = '/host/data_hdd/ctc/ss/example/saves/prediction_model.hdf5'
label_mappings = '/host/data_hdd/ctc/ss/example/saves/label_mapping.json'
deploy_path = '/host/data_hdd/ctc/ss/example/deploy/'
deploy_version = 1

sess = tf.Session()

# Missing this was the source of one of the most challenging an insidious bugs that I've ever encountered.
# Without explicitly linking the session the weights for the dense layer added below don't get loaded
# and so the model returns random results which vary with each model you upload because of random seeds.
K.set_session(sess)


model = load_model(model_path)
model_json = model.to_json()
model_weights = model.get_weights()

K._LEARNING_PHASE = tf.constant(0)
K.set_learning_phase(0)


new_model = model_from_json(model_json)
new_model.set_weights(model_weights)


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




##################################
# OLD
##################################

model = load_model(model_path)


# Build new model for serialization

K.set_learning_phase(0)  # all new operations will be in test mode from now on
# serialize the model and get its weights, for quick re-building
model_json = model.to_json()
model_weights = model.get_weights()

# re-build a model where the learning phase is now hard-coded to 0
new_model = model_from_json(model_json)
new_model.set_weights(model_weights)



saver = tf.train.Saver(sharded=True)
model_exporter = exporter.Exporter(saver)
signature = exporter.classification_signature(input_tensor=model.input,
                                              scores_tensor=model.output)

model_exporter.init(sess.graph.as_graph_def(),
                    default_graph_signature=signature)
model_exporter.export(deploy_path, tf.constant(deploy_version), sess)
