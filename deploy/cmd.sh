sudo rm -r ~/code/camera-trap-classifier
git clone -b deploy_models https://github.com/marco-willi/camera-trap-classifier.git ~/code/camera-trap-classifier


sudo docker run -it -v ~/:/host tensorflow/tensorflow:1.9.0-rc1-devel-py3 bash


#########################
# Estimator Serving
#########################

# INFO:tensorflow:Calling model_fn.
# INFO:tensorflow:Done calling model_fn.
# INFO:tensorflow:Signatures INCLUDED in export for Predict: ['serving_default']
# INFO:tensorflow:Signatures INCLUDED in export for Classify: None
# INFO:tensorflow:Signatures INCLUDED in export for Eval: None
# INFO:tensorflow:Signatures INCLUDED in export for Regress: None
# INFO:tensorflow:Signatures INCLUDED in export for Train: None
# INFO:tensorflow:Restoring parameters from /host/data_hdd/ctc/ss/example/estimator/keras_model.ckpt
# INFO:tensorflow:Assets added to graph.
# INFO:tensorflow:No assets to write.
# INFO:tensorflow:SavedModel written to: /host/data_hdd/ctc/ss/example/deploy_estimator/temp-b'1529270278'/saved_model.pb
# b'/host/data_hdd/ctc/ss/example/deploy_estimator/1529270278'
# >>> exit()
# root@770eaf97d891:/host/code/camera-trap-classifier# python3



bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server  \
--port=3000 \
--model_name=keras_model \
--model_base_path=/host/data_hdd/ctc/ss/example/deploy_estimator &> estimator &




################
# Protobufs
################
# note tensorflow_original instead of tensorflow
cd /host/code/serving
python -m grpc.tools.protoc ./tensorflow_original/tensorflow/core/framework/*.proto \
--python_out=/host/code/camera-trap-classifier/ \
--grpc_python_out=/host/code/camera-trap-classifier/ --proto_path=.



################
# Test Client
################
pip install tensorflow-serving-api-python3
