# Test Cats vs Dogs

# Read from Class Directories
python create_dataset_inventory.py dir -path D:\\Studium_GD\\Zooniverse\\Data\\transfer_learning_project\\images\\4715\\all \
-export_path ./test_big/cat_dog_dir_test.json

# Create TFRecord Files
python create_dataset.py -inventory ./test_big/cat_dog_dir_test.json \
-output_dir ./test_big/cats_vs_dogs/tfr_files/ \
-image_save_side_max 200 \
-split_percent 0.7 0.15 0.15 \
-overwrite

# Train a Model
python main_train_new.py \
-train_tfr_path ./test_big/cats_vs_dogs/tfr_files \
-train_tfr_prefix train \
-val_tfr_path ./test_big/cats_vs_dogs/tfr_files \
-val_tfr_prefix val \
-test_tfr_path ./test_big/cats_vs_dogs/tfr_files \
-test_tfr_prefix test \
-class_mapping_json ./test_big/cats_vs_dogs/tfr_files/label_mapping.json \
-run_outputs_dir ./test_big/cats_vs_dogs/run_outputs/ \
-model_save_dir ./test_big/cats_vs_dogs/model_save_dir/ \
-model cats_vs_dogs \
-labels class \
-batch_size 128 \
-n_cpus 2 \
-n_gpus 1 \
-buffer_size 512 \
-max_epochs 10 \
-starting_epoch 0



python main_train_new.py \
-train_tfr_path ./test_big/cats_vs_dogs/tfr_files \
-train_tfr_prefix train \
-val_tfr_path ./test_big/cats_vs_dogs/tfr_files \
-val_tfr_prefix val \
-test_tfr_path ./test_big/cats_vs_dogs/tfr_files \
-test_tfr_prefix test \
-class_mapping_json ./test_big/cats_vs_dogs/tfr_files/label_mapping.json \
-run_outputs_dir ./test_big/cats_vs_dogs/run_outputs/ \
-model_save_dir ./test_big/cats_vs_dogs/model_save_dir/ \
-model cats_vs_dogs \
-labels class \
-batch_size 128 \
-n_cpus 2 \
-n_gpus 1 \
-buffer_size 512 \
-max_epochs 6 \
-starting_epoch 6 \
-continue_training \
-model_to_load ./test_big/cats_vs_dogs/run_outputs/model_epoch_5.hdf5
