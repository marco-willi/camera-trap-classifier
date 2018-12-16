""" Process Individual Predictions
    - Extract
    - Aggregate
"""


class ProcessPredictions(object):
    def __init__(self, model_outputs, id_to_class_mapping):
        self.outputs = model_outputs
        self.id_to_class_mapping = id_to_class_mapping
        self.output_to_pretty = \
            {x: x.split('label/')[-1] for x in self.outputs}
        self.id_to_class_mapping_clean = {'label/' + k: v for k, v in
                                          self.id_to_class_mapping.items()}

    def map_and_extract_model_prediction(self, preds):
        """ Process a single prediction """
        result = dict()

        # Loop over all labels (self.outputs)
        for i, output in enumerate(self.outputs):

            all_numeric_outputs = \
                list(self.id_to_class_mapping_clean[output].keys())
            all_numeric_outputs.sort()

            # extract predictions for each label of the current output
            preds_for_output = preds[i]

            all_class_preds = [preds_for_output[x] for x in
                               all_numeric_outputs]

            all_class_preds_mapped = {
                    self.id_to_class_mapping_clean[output][i]: pred for i, pred
                    in enumerate(all_class_preds)
                    }

            result[self.output_to_pretty[output]] = all_class_preds_mapped

        return result

    def map_and_extract_ground_truth(self, labels):
        """ Process a single prediction """
        result = dict()

        # Loop over all labels (self.outputs)
        for i, output in enumerate(self.outputs):

            truth = self._try_extracting_ground_truth(labels, output)

            if truth is not None:
                result[self.output_to_pretty[output]] = truth

        if len(result.keys()) == 0:
            return None
        return result

    def _try_extracting_ground_truth(self, labels, output):
        """ Try to find and extract ground truth """
        # try to find the output in the features
        try:
            truth_for_output = labels[output]
            if isinstance(truth_for_output, (int, float)):
                try:
                    truth_numeric = int(truth_for_output)
                    truth_mapped = \
                        self.id_to_class_mapping_clean[output][truth_numeric]
                    return truth_mapped
                except:
                    pass
            truth_mapped = truth_for_output.decode('utf-8')
        except:
            return None
        return truth_mapped

    def process_predictions(self, predictions, aggregation_mode='mean'):
        """ Process a batch of predictions """
        for _id, data in predictions.items():
            preds_list = [x['predictions'] for x in data['images']]
            # collect all predictions
            collected = self._collect_predictions(preds_list)
            # consolidate predictions
            consolidated = self._consolidate_predictions(collected)
            # aggregate predictions
            aggregated = self._aggregate_predictions(
                consolidated,
                aggregation_mode)
            # get top predictions
            top_preds, top_confs = self._get_top_predictions(aggregated)

            # add info
            data['aggregated_pred'] = aggregated
            data['predictions_top'] = top_preds
            data['confidences_top'] = top_confs

        return predictions

    def _collect_predictions(self, extracted_predictions):
        """ Collect all predictions  """
        for i, image_pred in enumerate(extracted_predictions):
            all_label_names = image_pred.keys()
            if i == 0:
                preds_per_label = {x: list() for x in all_label_names}
            for label in all_label_names:
                preds_of_image_and_label = image_pred[label]
                preds_per_label[label].append(preds_of_image_and_label)
        return preds_per_label

    def _consolidate_predictions(self, collected_preds):
        """ Consolidate Predictions """

        label_names = collected_preds.keys()
        consolidated = {k: {} for k in label_names}
        for label_name in label_names:
            label_pred_list = collected_preds[label_name]
            for label_pred in label_pred_list:
                for label, pred in label_pred.items():
                    if label not in consolidated[label_name]:
                        consolidated[label_name][label] = list()
                    consolidated[label_name][label].append(pred)
        return consolidated

    def _aggregate_predictions(self, consolidated_predictions, mode='mean'):
        """ Aggregate Predictions of Multiple Images / ID

        """
        agg_label = dict()
        for label_name, labels in consolidated_predictions.items():
            agg_label[label_name] = dict()
            for label, preds_list in labels.items():
                if mode == 'mean':
                    agg = sum([float(x) for x in preds_list]) / len(preds_list)
                elif mode == 'max':
                    agg = max([float(x) for x in preds_list])
                elif mode == 'min':
                    agg = min([float(x) for x in preds_list])
                else:
                    raise NotImplementedError(
                        "Aggregation mode %s not implemented" % mode)
                agg_label[label_name][label] = agg
        return agg_label

    def _get_top_predictions(self, predictions):
        """ Get top prediction for each label """
        top_preds = dict()
        top_confs = dict()
        for label_name, label_vals in predictions.items():

            ordered_classes = sorted(label_vals,
                                     key=label_vals.get,
                                     reverse=True)
            top_label = ordered_classes[0]
            top_value = label_vals[top_label]
            top_preds[label_name] = top_label
            top_confs[label_name] = top_value
        return top_preds, top_confs
