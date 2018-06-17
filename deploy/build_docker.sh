#pip install tensorflow-serving-api-python3
# https://www.tensorflow.org/serving/docker

####################
cd ~/code
git clone https://github.com/tensorflow/serving.git
cd ./serving/tensorflow_serving/tools/docker/
sudo docker build --pull -t $USER/tensorflow-serving-devel-cpu -f Dockerfile.devel .
sudo docker run -it -p 9000:9000 $USER/tensorflow-serving-devel-cpu /bin/bash

 cd tensorflow_serving/
 bazel build -c opt --copt=-msse4.1 --copt=-msse4.2 tensorflow_serving/...
# INFO: Elapsed time: 7631.549s, Critical Path: 101.76s

# verify if server is running (in /tensorflow-serving)
bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server

# exit the container and commit changes
# sudo docker ps (to get container id)
sudo docker commit container_id $USER/tensorflow-serving-devel-cpu

#######################
# Start Service
#######################
sudo docker run -it -d -P -v ~/:/host --name tf_serving_cpu -p 3000:3000 $USER/tensorflow-serving-devel-cpu
sudo docker attach tf_serving_cpu


# Make sure the model is saved in
# /host/data_hdd/ctc/ss/example/deploy/1/..

bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server  \
--port=3000 \
--model_name=class \
--model_base_path=/host/data_hdd/ctc/ss/example/deploy &> class &
