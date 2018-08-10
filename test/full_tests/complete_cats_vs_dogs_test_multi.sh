# Test Multi-Image Captures
# ./test/full_tests/complete_cats_vs_dogs_test_multi.sh
image_root_path=D:\\Studium_GD\\Zooniverse\\Data\\transfer_learning_project\\images\\4715\\all
export_root_path=./test_big//cats_vs_dogs_multi/
image_root_path=./test/test_images/
tfr_files_path=${export_root_path}tfr_files/
run_outputs_path=${export_root_path}run_outputs/
model_save_dir=${export_root_path}model_save_dir/
run_outputs_tl_path=${export_root_path}run_outputs_tl/
model_save_tl_dir=${export_root_path}model_save_dir_tl/

# delete files
rm $tfr_files_path/*
rm $model_save_dir/*.hdf5
rm $run_outputs_path/*
rm $run_outputs_tl_path/*

# Read from csv
python create_dataset_inventory.py csv -path ./test/test_files/cats_vs_dogs_multi.csv \
-export_path ${export_root_path}inventory.json \
-capture_id_field id \
-image_fields image \
-label_fields species standing count

# Create TFRecord Files
python create_dataset.py -inventory ${export_root_path}inventory.json \
-output_dir ${tfr_files_path} \
-image_save_side_max 200 \
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
-n_gpus 1 \
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
-n_gpus 1 \
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
