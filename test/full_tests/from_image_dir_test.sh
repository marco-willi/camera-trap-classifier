# Test from image path
# ./test/full_tests/from_image_dir_test.sh
image_root_path=D:\\Studium_GD\\Zooniverse\\Data\\transfer_learning_project\\images\\4715\\all
export_root_path=./test_big/cats_vs_dogs/
tfr_files_path=${export_root_path}tfr_files/
run_outputs_path=${export_root_path}run_outputs/
model_save_dir=${export_root_path}model_save_dir/
estimator_deploy_save_dir=${model_save_dir}estimator_deploy/
estimator_save_dir=${model_save_dir}estimator/
run_outputs_tl_path=${export_root_path}run_outputs_tl/
model_save_tl_dir=${export_root_path}model_save_dir_tl/
run_outputs_ft_path=${export_root_path}run_outputs_ft/
model_save_ft_dir=${export_root_path}model_save_dir_ft/

# delete files
rm $tfr_files_path/*
rm $model_save_dir/*.hdf5
rm $run_outputs_path/*
rm $run_outputs_tl_path/*
rm $run_outputs_ft_path/*

# Read from Class Directories
python create_dataset_inventory.py dir -path $image_root_path \
-export_path ${export_root_path}inventory.json

# Create TFRecord Files
python create_dataset.py -inventory ${export_root_path}inventory.json \
-output_dir ${tfr_files_path} \
-image_save_side_max 200 \
-split_percent 0.7 0.15 0.15 \
-overwrite \
-process_images_in_parallel \
-process_images_in_parallel_size 2000 \
-processes_images_in_parallel_n_processes 2

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
-labels class \
-batch_size 128 \
-n_cpus 2 \
-n_gpus 1 \
-buffer_size 512 \
-max_epochs 6 \
-starting_epoch 0


# Continue Training
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
-labels class \
-batch_size 128 \
-n_cpus 2 \
-n_gpus 1 \
-buffer_size 512 \
-max_epochs 7 \
-starting_epoch 6 \
-continue_training \
-model_to_load ${run_outputs_path} \
-color_augmentation full_fast


# Pseudo Transfer Training
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
-labels class \
-batch_size 128 \
-n_cpus 2 \
-n_gpus 1 \
-buffer_size 512 \
-max_epochs 4 \
-starting_epoch 0 \
-transfer_learning \
-model_to_load ${run_outputs_path} \
-ignore_aspect_ratio


# Pseudo Fine Tuning
python train.py \
-train_tfr_path ${tfr_files_path} \
-train_tfr_pattern train \
-val_tfr_path ${tfr_files_path} \
-val_tfr_pattern val \
-test_tfr_path ${tfr_files_path} \
-test_tfr_pattern test \
-class_mapping_json ${tfr_files_path}label_mapping.json \
-run_outputs_dir ${run_outputs_ft_path} \
-model_save_dir ${model_save_ft_dir} \
-model small_cnn \
-labels class \
-batch_size 128 \
-n_cpus 2 \
-n_gpus 1 \
-buffer_size 512 \
-max_epochs 4 \
-starting_epoch 0 \
-transfer_learning \
-transfer_learning_type all_layers \
-model_to_load ${run_outputs_path}


# Deploy model
python export.py -model ${model_save_dir}best_model.hdf5 \
-class_mapping_json ${model_save_dir}label_mappings.json \
-pre_processing_json ${model_save_dir}image_processing.json \
-output_dir ${estimator_deploy_save_dir} \
-estimator_save_dir ${estimator_save_dir}


# Create Predictions
python predict.py \
-image_dir ${image_root_path} \
-results_file ${model_save_dir}preds.csv \
-model_path ${model_save_dir}best_model.hdf5 \
-class_mapping_json ${model_save_dir}label_mappings.json \
-pre_processing_json ${model_save_dir}image_processing.json
