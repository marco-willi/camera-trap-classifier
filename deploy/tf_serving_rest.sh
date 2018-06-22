# Configure a TF-Serving REST API

# Get code and Dockerfile
git clone https://github.com/tensorflow/serving.git
cd ./serving/tensorflow_serving/tools/docker/

# OPTIONAL: Optimized build - Use Dockerfile.devel and adjust compile flags
# bazel build -c opt --copt=-msse4.1 --copt=-msse4.2 tensorflow_serving/...
# sudo docker build --pull -t $USER/tensorflow-serving-devel-cpu -f Dockerfile.devel .

sudo docker build --pull -t $USER/tensorflow-serving-cpu -f Dockerfile .

#######################
# Start Service
#######################

# start container and expose REST API
sudo docker run -it -d -P --name tf_serving_cpu -p 8501:8501 $USER/tensorflow-serving-cpu

# create a model directory inside the container
sudo docker exec -d tf_serving_cpu mkdir /models/model

# copy a model into the container
sudo docker cp ~/data_hdd/ctc/ss/example/deploy_estimator/1529626634 tf_serving_cpu:/models/model/
