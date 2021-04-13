python src/validate_on_lfw.py \
../datasets/crop7_test \
../models/20210407-203650 \
--lfw_pairs ../datasets/crop7_test/pairs.txt \
--distance_metric 1 \
--subtract_mean \
--lfw_batch_size 6 \
--lfw_nrof_folds 2 \
--image_size 100 \