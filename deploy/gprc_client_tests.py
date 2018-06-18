""" https://raw.githubusercontent.com/yu-iskw/tensorflow-serving-example/master/python/grpc_client.py """
from __future__ import print_function
import argparse
import time
import numpy as np
from PIL import Image
import io
from grpc.beta import implementations
from tensorflow.contrib.util import make_tensor_proto
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2


def read_jpeg(image):
    """ Reads jpeg and returns Bytes """
    img = Image.open(image)
    b = io.BytesIO()
    img.save(b, 'JPEG')
    image_bytes = b.getvalue()
    return image_bytes


channel = implementations.insecure_channel('0.0.0.0', 3000)
stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
example_image = '/host/data_hdd/ctc/ss/images/elephant/ASG000r7uh_0.jpeg'
data = read_jpeg(example_image)
request = predict_pb2.PredictRequest()
request.model_spec.name = 'keras_model'
request.model_spec.signature_name = 'serving_default'
request.inputs['image'].CopyFrom(make_tensor_proto(data))
result = stub.Predict(request, 10.0)



def run(host, port, image, model, signature_name):

    # channel = grpc.insecure_channel('%s:%d' % (host, port))


    # Read an image
    data = read_jpeg(image)

    start = time.time()

    # Call classification model to make prediction on the image
    request = predict_pb2.PredictRequest()
    request.model_spec.name = model
    request.model_spec.signature_name = signature_name
    request.inputs['image'].CopyFrom(make_tensor_proto(data))

    result = stub.Predict(request, 10.0)

    end = time.time()
    time_diff = end - start

    # Reference:
    # How to access nested values
    # https://stackoverflow.com/questions/44785847/how-to-retrieve-float-val-from-a-predictresponse-object
    print(result)
    print('time elapased: {}'.format(time_diff))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', help='Tensorflow server host name', default='localhost', type=str)
    parser.add_argument('--port', help='Tensorflow server port number', default=8500, type=int)
    parser.add_argument('--image', help='input image', type=str)
    parser.add_argument('--model', help='model name', type=str)
    parser.add_argument('--signature_name', help='Signature name of saved TF model',
                        default='serving_default', type=str)

    args = parser.parse_args()
    run(args.host, args.port, args.image, args.model, args.signature_name)
