pip install tensorflow-serving-api-python3

wget -O https://github.com/tensorflow/serving/blob/master/tensorflow_serving/tools/docker/Dockerfile.devel

docker build --pull -t $USER/tensorflow-serving-devel -f Dockerfile.devel .
docker run -it $USER/tensorflow-serving-devel

git clone --recurse-submodules https://github.com/tensorflow/serving
cd serving/tensorflow
./configure
cd ..
bazel test tensorflow_serving/...
