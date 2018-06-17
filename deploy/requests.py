""" Create Classification Requests """

# outside docker
pip install grpcio grpcio-tools

cd ~/code
mv tensorflow/ serving/

python -m grpc.tools.protoc ./tensorflow_serving/apis/*.proto \
--python_out= /home/ubuntu/code/test \
--grpc_python_out= /home/ubuntu/code/test \
--proto_path=.


python -m grpc.tools.protoc ./tensorflow_serving/apis/*.proto \
--python_out= ~/data_hdd/ctc/ss/example/deploy/1 \
--grpc_python_out= ~/data_hdd/ctc/ss/example/deploy/1 \
--proto_path=.
