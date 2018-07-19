# Test Cats vs Dogs

# Read from csv
python create_dataset_inventory.py csv -path ./test/test_files/cats_vs_dogs_multi.csv \
-export_path ./test_big/cats_vs_dogs_multi/inventory.json \
-capture_id_field id \
-image_fields image \
-label_fields species standing count

# Create TFRecord Files
python create_dataset.py -inventory ./test_big/cats_vs_dogs_multi/inventory.json \
-output_dir ./test_big/cats_vs_dogs_multi/tfr_files/ \
-image_save_side_max 200 \
-split_percent 0.7 0.15 0.15 \
-overwrite

# Train a Model
python train.py \
-train_tfr_path ./test_big/cats_vs_dogs_multi/tfr_files \
-train_tfr_prefix train \
-val_tfr_path ./test_big/cats_vs_dogs_multi/tfr_files \
-val_tfr_prefix val \
-test_tfr_path ./test_big/cats_vs_dogs_multi/tfr_files \
-test_tfr_prefix test \
-class_mapping_json ./test_big/cats_vs_dogs_multi/tfr_files/label_mapping.json \
-run_outputs_dir ./test_big/cats_vs_dogs/run_outputs/ \
-model_save_dir ./test_big/cats_vs_dogs/model_save_dir/ \
-model small_cnn \
-labels species standing \
-batch_size 12 \
-n_cpus 2 \
-n_gpus 1 \
-buffer_size 1 \
-max_epochs 70 \
-starting_epoch 0

# Transfer Learning
python train.py \
-train_tfr_path ./test_big/cats_vs_dogs_multi/tfr_files \
-train_tfr_prefix train \
-val_tfr_path ./test_big/cats_vs_dogs_multi/tfr_files \
-val_tfr_prefix val \
-test_tfr_path ./test_big/cats_vs_dogs_multi/tfr_files \
-test_tfr_prefix test \
-class_mapping_json ./test_big/cats_vs_dogs_multi/tfr_files/label_mapping.json \
-run_outputs_dir ./test_big/cats_vs_dogs_multi/run_outputs_tl/ \
-model_save_dir ./test_big/cats_vs_dogs_multi/model_save_dir_tl/ \
-model cats_vs_dogs \
-labels species standing \
-batch_size 12 \
-n_cpus 2 \
-n_gpus 1 \
-buffer_size 1 \
-max_epochs 70 \
-starting_epoch 0 \
-transfer_learning \
-model_to_load ./test_big/cats_vs_dogs/run_outputs/
