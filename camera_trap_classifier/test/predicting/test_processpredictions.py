""" Test ProcessPredictions Class """
import unittest

from camera_trap_classifier.predicting.processor import ProcessPredictions


class ProcessPredictionsTestsr(unittest.TestCase):
    """ Test Processing of PRedictions """

    def setUp(self):
        model_outputs = ['label/species', 'label/counts']
        mapping = {
            'species': {0: 'cat',  1: 'dog'},
            'counts': {0: '1', 1: '2', 2: '3', 3: '4'}}
        self.example = {
             '1234':{
             'images': [{
                     'path': 'image1',
                     'predictions': {
                        'species': {'cats': 0.8, 'dogs': 0.6},
                        'counts': {'1': 0.5, '2': 0.5}
                        }
                     },{
                     'path': 'image2',
                     'predictions':{
                        'species': {'cats': 0.2, 'dogs': 0.8},
                        'counts': {'1': 0.6, '2': 0.4}
                        }
                     }],
             'meta_data': {'timestamp': 'lala'},
             'ground_truth': {
              'species': 'cats',
              'counts': 'ClassB'
              }},
             '1235':{
             'images': [{
                     'path': 'image1',
                     'predictions': {
                        'species': {'cats': 0.5, 'dogs': 0.5},
                        'counts': {'1': 0.5, '2': 0.5}
                        }
                     },{
                     'path': 'image2',
                     'predictions':{
                        'species': {'cats': 0.2, 'dogs': 0.8},
                        'counts': {'1': 0.6, '2': 0.4}
                        }
                     }],
             'meta_data': {'timestamp': 'lala'},
             'ground_truth': {
              'species': 'cats',
              'counts': 'ClassB'
              }},
             '1236':{
             'images': [{
                     'path': 'image1',
                     'predictions': {
                        'counts': {'1': 0.5, '2': 0.5}
                        }
                     },{
                     'path': 'image2',
                     'predictions':{
                        'species': {'cats': 0.2, 'dogs': 0.8},
                        'counts': {'1': 0.6, '2': 0.4}
                        }
                     }],
             'meta_data': {'timestamp': 'lala'},
             'ground_truth': {
              'species': 'cats',
              'counts': 'ClassB'
              }}
            }
        self.processor = ProcessPredictions(
                            model_outputs=model_outputs,
                            id_to_class_mapping=mapping)

    def testMapandExtractPredictionStandard(self):
        input = [[0.3, 0.7], [0.1, 0.1, 0.2, 0.6]]
        expected = {
            'species': {'cat': 0.3, 'dog': 0.7},
            'counts': {'1': 0.1, '2': 0.1, '3': 0.2, '4': 0.6}}
        actual = self.processor.map_and_extract_model_prediction(input)
        self.assertEqual(expected, actual)

    def testMapandExtractGroundTruth(self):
        input = {'label/species': b'cat', 'label/counts': b'4'}
        expected = {'species': 'cat', 'counts': '4'}
        actual = self.processor.map_and_extract_ground_truth(input)
        self.assertEqual(expected, actual)

    def testMapandExtractGroundTruthInt(self):
        input = {'label/species': b'cat', 'label/counts': 3}
        expected = {'species': 'cat', 'counts': '4'}
        actual = self.processor.map_and_extract_ground_truth(input)
        self.assertEqual(expected, actual)

    def testCollectPredictionsStandard(self):
        input = [{'species': {'cats': 0.5, 'dogs': 0.5},
                  'counts': {'1': 0.5, '2': 0.5}},
                 {'species': {'cats': 0.2, 'dogs': 0.8},
                  'counts': {'1': 0.6, '2': 0.4}}]
        expected = {'species': [{'cats': 0.5, 'dogs': 0.5}, {'cats': 0.2, 'dogs': 0.8}],
                    'counts': [{'1': 0.5, '2': 0.5}, {'1': 0.6, '2': 0.4}]}
        actual = self.processor._collect_predictions(input)
        self.assertEqual(expected, actual)

    def testCollectPredictionsWithMissing(self):
        input = [{'counts': {'1': 0.5, '2': 0.5}},
                 {'species': {'cats': 0.2, 'dogs': 0.8},
                  'counts': {'1': 0.6, '2': 0.4}}]
        expected = {'species': [{'cats': 0.2, 'dogs': 0.8}],
                    'counts': [{'1': 0.5, '2': 0.5}, {'1': 0.6, '2': 0.4}]}
        actual = self.processor._collect_predictions(input)
        self.assertEqual(expected, actual)

    def testCollectPredictionsWithAllMissing(self):
        input = [{},
                 {'species': {'cats': 0.2, 'dogs': 0.8},
                  'counts': {'1': 0.6, '2': 0.4}}]
        expected = {'species': [{'cats': 0.2, 'dogs': 0.8}],
                    'counts': [{'1': 0.6, '2': 0.4}]}
        actual = self.processor._collect_predictions(input)
        self.assertEqual(expected, actual)

    def testConsolidatePredictionsStandard(self):
        input = {'species': [{'cats': 0.5, 'dogs': 0.5}, {'cats': 0.2, 'dogs': 0.8}],
                    'counts': [{'1': 0.5, '2': 0.5}, {'1': 0.6, '2': 0.4}]}

        expected = {'species': {'cats': [0.5, 0.2], 'dogs': [0.5, 0.8]},
                    'counts': {'1': [0.5, 0.6], '2': [0.5, 0.4]}}

        actual = self.processor._consolidate_predictions(input)
        self.assertEqual(expected, actual)

    def testAggregatePredictionsMean(self):
        input = {'species': {'cats': [0.5, 0.2], 'dogs': [0.5, 0.8]},
                    'counts': {'1': [0.5, 0.6], '2': [0.5, 0.4]}}
        expected = {'species': {'cats': 0.35, 'dogs': 0.65},
                    'counts': {'1': 0.55, '2': 0.45}}
        actual = self.processor._aggregate_predictions(input, mode='mean')
        self.assertEqual(expected, actual)

    def testAggregatePredictionsMax(self):
        input = {'species': {'cats': [0.5, 0.2], 'dogs': [0.5, 0.8]},
                    'counts': {'1': [0.5, 0.6], '2': [0.5, 0.4]}}
        expected = {'species': {'cats': 0.5, 'dogs': 0.8},
                    'counts': {'1': 0.6, '2': 0.5}}
        actual = self.processor._aggregate_predictions(input, mode='max')
        self.assertEqual(expected, actual)

    def testAggregatePredictionsMin(self):
        input = {'species': {'cats': [0.5, 0.2], 'dogs': [0.5, 0.8]},
                    'counts': {'1': [0.5, 0.6], '2': [0.5, 0.4]}}
        expected = {'species': {'cats': 0.2, 'dogs': 0.5},
                    'counts': {'1': 0.5, '2': 0.4}}
        actual = self.processor._aggregate_predictions(input, mode='min')
        self.assertEqual(expected, actual)

    def testProcessPredictionsStandard(self):
        expected = \
        {'1234': {
            'images': [
                {'path': 'image1',
                 'predictions': {'species': {'cats': 0.8, 'dogs': 0.6},
                                 'counts': {'1': 0.5, '2': 0.5}}},
                 {'path': 'image2',
                  'predictions': {'species': {'cats': 0.2, 'dogs': 0.8},
                                  'counts': {'1': 0.6, '2': 0.4}}}],
          'meta_data': {'timestamp': 'lala'},
          'ground_truth': {'species': 'cats', 'counts': 'ClassB'},
          'aggregated_pred':
              {'species': {'cats': 0.5, 'dogs': 0.7},
               'counts': {'1': 0.55, '2': 0.45}},
          'predictions_top':
              {'species': 'dogs',
               'counts': '1'},
        'confidences_top': {'species': 0.7, 'counts': 0.55}},
         '1235': {
            'images': [
                {'path': 'image1',
                 'predictions': {'species': {'cats': 0.5, 'dogs': 0.5},
                                 'counts': {'1': 0.5, '2': 0.5}}},
                 {'path': 'image2',
                  'predictions': {'species': {'cats': 0.2, 'dogs': 0.8},
                                  'counts': {'1': 0.6, '2': 0.4}}}],
            'meta_data': {'timestamp': 'lala'},
            'ground_truth': {'species': 'cats', 'counts': 'ClassB'},
            'aggregated_pred': {
                    'species': {'cats': 0.35, 'dogs': 0.65},
                    'counts': {'1': 0.55, '2': 0.45}},
          'predictions_top': {
                  'species': 'dogs',
                  'counts': '1'},
           'confidences_top': {'species': 0.65, 'counts': 0.55}},
         '1236': {
            'images': [
                {'path': 'image1',
                 'predictions': {
                                 'counts': {'1': 0.5, '2': 0.5}}},
                 {'path': 'image2',
                  'predictions': {'species': {'cats': 0.2, 'dogs': 0.8},
                                  'counts': {'1': 0.6, '2': 0.4}}}],
            'meta_data': {'timestamp': 'lala'},
            'ground_truth': {'species': 'cats', 'counts': 'ClassB'},
            'aggregated_pred': {
                    'species': {'cats': 0.2, 'dogs': 0.8},
                    'counts': {'1': 0.55, '2': 0.45}},
          'predictions_top': {
                  'species': 'dogs',
                  'counts': '1'},
           'confidences_top': {'species': 0.8, 'counts': 0.55}}
           }
        actual = self.processor.process_predictions(self.example, aggregation_mode='mean')
        self.assertAlmostEqual(expected, actual, delta=0.0001)

if __name__ == '__main__':

    unittest.main()
