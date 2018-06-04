# Test Cats vs Dogs

# Read from Class Directories
python create_dataset_inventory.py dir -path D:\\Studium_GD\\Zooniverse\\Data\\transfer_learning_project\\images\\4715\\all \
-export_path ./test_big/cat_dog_dir_test.json

# Create TFRecord Files
python create_dataset.py -inventory ./test_big/cat_dog_dir_test.json \
-output_dir ./test_big/cats_vs_dogs/ \
-overwrite

# Train a Model
python main_train_new.py \
-train_tfr ./test_big/cats_vs_dogs/train.tfrecord \
-val_tfr ./test_big/cats_vs_dogs/val.tfrecord \
-test_tfr ./test_big/cats_vs_dogs/test.tfrecord \
-class_mapping_json ./test_big/cats_vs_dogs/label_mapping.json \
-run_outputs_dir ./test_big/cats_vs_dogs/run_outputs/ \
-model_save_dir ./test_big/cats_vs_dogs/model_save_dir/ \
-model cats_vs_dogs \
-labels class \
-batch_size 128 \
-n_cpus 2 \
-n_gpus 1 \
-buffer_size 2048 \
-max_epochs 70 \
-starting_epoch 0
