python src/train_tripletloss.py \
--logs_base_dir ../logs/ \
--models_base_dir ../models/ \
--data_dir ../datasets/crop7_train \
--image_size 100 \
--model_def models.inception_resnet_v1 \
--optimizer RMSPROP \
--learning_rate 0.01 \
--weight_decay 1e-4 \

