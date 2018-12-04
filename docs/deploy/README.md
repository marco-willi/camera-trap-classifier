# Deploy a Model using Tensorflow-Serving

Deploying a model into production can be done with TensorFlow-Serving. This readme describes how to set up a
REST API and post data to an exported model. There are two different approaches to achieve that.

## Tensorflow-Serving REST API

Tensorflow-Serving provides a REST API out-of-the-box (https://www.tensorflow.org/serving/api_rest).

### Set-Up with Docker

 We can use docker to serve a model (tf_serving_rest.sh for more details). The following command builds a container using only CPUs for Tensorflow which is slower but much cheaper when making inference. A GPU version is available if required.

```
# Get code and Dockerfile
git clone https://github.com/tensorflow/serving.git
cd ./serving/tensorflow_serving/tools/docker/

sudo docker build --pull -t $USER/tensorflow-serving-cpu -f Dockerfile .

# start container and expose REST API
sudo docker run -it -d -P --name tf_serving_cpu -p 8501:8501 $USER/tensorflow-serving-cpu

# create a model directory inside the container
sudo docker exec -d tf_serving_cpu mkdir /models/model

# copy a model into the container
sudo docker cp ~/data_hdd/ctc/ss/example/deploy_estimator/1529626634 tf_serving_cpu:/models/model/
```

Note, the folder where the model resides has to be integer named (e.g. 1529626634). Whenever a new model is copied into the container (with a larger directory number), Tensorflow-Serving switches to that model.


### Example Request Format

A request must follow this format (note that this won't actually work since the byte strings are cut off):

```
curl -X POST -d '{
"signature_name": "serving_default",
 "instances": [
{"image": {"b64":"/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgG...."}},
{"image": {"b64":"/9j/4AAQSkZJRgABAQAAAQABAAD/2..."}}]}' }}]}' http://localhost:8501/v1/models/model:predict
```

### Example Response Format

We have requested the predictions for two images. The response is a list of lists with two entries, each of which
has three entries because the model's output is three classes. The class mapping must be known.

```
{
    "predictions": [[0.333434, 0.509063, 0.157503], [0.333434, 0.509063, 0.157503]]
}
```

## Flask REST API

The downside of using the Tensorflow-Serving REST API is that it requires byte-strings as input. Coupling a separate API with Tensorflow-Serving allows for posting URLs. In this case we use Flask for posting a list
of URLs which are then being requested by the Flask server, served to the model, and subsequently the predictions are returned to the client.

### Set-Up with Docker-Compose

We refer to the following repository to set up the Flask API:

https://github.com/marco-willi/tf_serving_flask_app


## Example Request

The server can be tested by running this command on the running instance:

```
# Elephant and Zebra and Lion
curl -X POST -d '{"url": ["https://panoptes-uploads.zooniverse.org/production/subject_location/9e4556a3-5aba-46d0-a932-1f0e9e158d0d.jpeg", "https://s3-eu-west-1.amazonaws.com/pantherabucket1/17_2013/CS38_40185_20130427_101803.jpg","https://static.zooniverse.org/www.snapshotserengeti.org/subjects/standard/50c213e88a607540b9033aed_0.jpg"]}' -H 'Content-Type: application/json' http://0.0.0.0:5000/tf_api/species_client/prediction
```

## Example Response

This is an example response. It would be possible to incorporate the real class-mappings of the model - instead of 'class': 1 it would be 'species': 'Elephant' for example.

```
{
    "prediction_result": [
        [
            {
                "prob": 0.9343334436416626,
                "class": 1
            },
            {
                "prob": 0.06039385870099068,
                "class": 0
            },
            {
                "prob": 0.005272689741104841,
                "class": 2
            }
        ],
        [
            {
                "prob": 0.9999719858169556,
                "class": 2
            },
            {
                "prob": 1.4948362149880268e-05,
                "class": 1
            },
            {
                "prob": 1.306723333982518e-05,
                "class": 0
            }
        ],
        [
            {
                "prob": 0.6617377996444702,
                "class": 0
            },
            {
                "prob": 0.24102909862995148,
                "class": 2
            },
            {
                "prob": 0.09723305702209473,
                "class": 1
            }
        ]
    ]
}
```
