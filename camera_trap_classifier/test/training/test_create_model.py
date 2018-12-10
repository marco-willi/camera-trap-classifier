import tensorflow as tf

from camera_trap_classifier.training.prepare_model import create_model
from camera_trap_classifier.data.utils import generate_synthetic_data
from camera_trap_classifier.config.config import ConfigLoader


class CreateModelTests(tf.test.TestCase):
    """ Test Create Model """

    def setUp(self):
        self.model_cfg = ConfigLoader('./config/config.yaml')
        self.models = list(self.model_cfg.cfg['models'].keys())
        self.labels = ['label/species', 'label/counts']
        self.labels_clean = ['label/%s' % l for l in self.labels]
        self.n_classes = [10, 3]
        self.n_images = 1
        self.batch_size = 4

    def testModelRuns(self):
        for model_name in self.models:
            m_cfg = self.model_cfg.cfg['models'][model_name]
            h = m_cfg['image_processing']['output_height']
            w = m_cfg['image_processing']['output_width']
            print("Testing Model: %s" % model_name)

            model = create_model(
                  model_name=model_name,
                  input_shape=(h, w, 3),
                  target_labels=self.labels_clean,
                  n_classes_per_label_type=self.n_classes,
                  n_gpus=0,
                  continue_training=False,
                  rebuild_model=False,
                  transfer_learning=False,
                  transfer_learning_type=None,
                  path_of_model_to_load=None,
                  initial_learning_rate=0.1,
                  output_loss_weights=None)
            callbacks_list = []

            def input_feeder():
                dataset = generate_synthetic_data(
                    batch_size=self.batch_size,
                    image_shape=(h, w, 3),
                    labels=self.labels_clean,
                    n_classes=self.n_classes,
                    n_images=1)
                return dataset

            model.fit(
              input_feeder(),
              epochs=2,
              steps_per_epoch=1,
              validation_data=input_feeder(),
              validation_steps=1,
              callbacks=callbacks_list,
              initial_epoch=0)

            preds = model.predict(input_feeder(), steps=1)

            for i, n_class in enumerate(self.n_classes):
                self.assertEqual(preds[i].shape, (self.batch_size, n_class))
