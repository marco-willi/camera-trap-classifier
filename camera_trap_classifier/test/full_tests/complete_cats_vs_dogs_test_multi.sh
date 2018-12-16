# Test Multi-Image Captures
# ./test/full_tests/complete_cats_vs_dogs_test_multi.sh
export_root_path=./test_big/cats_vs_dogs_multi/
image_root_path=./test/test_images/
test_files=./test/test_files/
tfr_files_path=${export_root_path}tfr_files/
run_outputs_path=${export_root_path}run_outputs/
model_save_dir=${export_root_path}model_save_dir/
run_outputs_tl_path=${export_root_path}run_outputs_tl/
model_save_tl_dir=${export_root_path}model_save_dir_tl/
run_outputs_gs_path=${export_root_path}run_outputs_gs/
model_save_gs_dir=${export_root_path}model_save_dir_gs/

# delete files
rm $tfr_files_path*
rm $model_save_dir*.hdf5
rm $run_outputs_path*
rm $run_outputs_tl_path*
rm $run_outputs_gs_path*
rm $model_save_gs_dir*

# Read from csv
python create_dataset_inventory.py csv -path ./test/test_files/cats_vs_dogs_multi.csv \
-export_path ${export_root_path}inventory.json \
-capture_id_field id \
-image_fields image \
-label_fields species standing count

# Create TFRecord Files
python create_dataset.py -inventory ${export_root_path}inventory.json \
-output_dir ${tfr_files_path} \
-image_save_side_smallest 200 \
-split_percent 0.7 0.15 0.15 \
-overwrite

# Train a Model
python train.py \
-train_tfr_path ${tfr_files_path} \
-train_tfr_pattern train \
-val_tfr_path ${tfr_files_path} \
-val_tfr_pattern val \
-test_tfr_path ${tfr_files_path} \
-test_tfr_pattern test \
-class_mapping_json ${tfr_files_path}label_mapping.json \
-run_outputs_dir ${run_outputs_path} \
-model_save_dir ${model_save_dir} \
-model small_cnn \
-labels species standing count \
-labels_loss_weights 1 0.2 0.5 \
-batch_size 12 \
-n_cpus 2 \
-n_gpus 0 \
-buffer_size 1 \
-max_epochs 70 \
-starting_epoch 0

# Transfer Learning
python train.py \
-train_tfr_path ${tfr_files_path} \
-train_tfr_pattern train \
-val_tfr_path ${tfr_files_path} \
-val_tfr_pattern val \
-test_tfr_path ${tfr_files_path} \
-test_tfr_pattern test \
-class_mapping_json ${tfr_files_path}label_mapping.json \
-run_outputs_dir ${run_outputs_tl_path} \
-model_save_dir ${model_save_tl_dir} \
-model small_cnn \
-labels species standing \
-batch_size 12 \
-n_cpus 2 \
-n_gpus 0 \
-buffer_size 1 \
-max_epochs 70 \
-starting_epoch 0 \
-transfer_learning \
-model_to_load ${run_outputs_path}

# Create Predictions
python predict.py \
-image_dir ${image_root_path} \
-results_file ${model_save_dir}preds.csv \
-model_path ${model_save_dir}best_model.hdf5 \
-class_mapping_json ${model_save_dir}label_mappings.json \
-pre_processing_json ${model_save_dir}image_processing.json


# Create Predictions from CSV
python predict.py \
-csv_path ${test_files}cats_vs_dogs_multi_image.csv \
-csv_id_col id \
-csv_images_cols image1 image2 image3 \
-results_file ${model_save_dir}preds_multi_mean.csv \
-model_path ${model_save_dir}best_model.hdf5 \
-class_mapping_json ${model_save_dir}label_mappings.json \
-pre_processing_json ${model_save_dir}image_processing.json \
-aggregation_mode mean


python predict.py \
-csv_path ${test_files}cats_vs_dogs_multi_image.csv \
-csv_id_col id \
-csv_images_cols image1 image2 image3 \
-results_file ${model_save_dir}preds_multi_max.csv \
-model_path ${model_save_dir}best_model.hdf5 \
-class_mapping_json ${model_save_dir}label_mappings.json \
-pre_processing_json ${model_save_dir}image_processing.json \
-aggregation_mode max


python predict.py \
-csv_path ${test_files}cats_vs_dogs_multi_image.csv \
-csv_id_col id \
-csv_images_cols image1 image2 image3 \
-export_file_type json \
-results_file ${model_save_dir}preds_multi_max.json \
-model_path ${model_save_dir}best_model.hdf5 \
-class_mapping_json ${model_save_dir}label_mappings.json \
-pre_processing_json ${model_save_dir}image_processing.json \
-aggregation_mode max



# Train a Model with grayscale stacking
python train.py \
-train_tfr_path ${tfr_files_path} \
-train_tfr_pattern train \
-val_tfr_path ${tfr_files_path} \
-val_tfr_pattern val \
-test_tfr_path ${tfr_files_path} \
-test_tfr_pattern test \
-class_mapping_json ${tfr_files_path}label_mapping.json \
-run_outputs_dir ${run_outputs_gs_path} \
-model_save_dir ${model_save_gs_dir} \
-model small_cnn \
-labels species \
-batch_size 12 \
-n_cpus 2 \
-n_gpus 0 \
-buffer_size 1 \
-max_epochs 2 \
-starting_epoch 0 \
-image_choice_for_sets grayscale_stacking


# Predict with grayscale_stacking
python predict.py \
-image_dir ${image_root_path} \
-results_file ${model_save_gs_dir}preds.csv \
-model_path ${model_save_gs_dir}best_model.hdf5 \
-class_mapping_json ${model_save_gs_dir}label_mappings.json \
-pre_processing_json ${model_save_gs_dir}image_processing.json
