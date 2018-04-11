# Prepare metadata (ok)
neptune run --config configs_end_to_end/neptune_size_estimator.yaml \
-- prepare_metadata --train_data --test_data

# Prepare masks (ok)
neptune run --config configs_end_to_end/neptune_size_estimator.yaml \
-- prepare_masks

# Train size estimator unet (ok)
neptune run --config configs_end_to_end/neptune_size_estimator.yaml \
-- train_pipeline --pipeline_name patched_unet_training --simple_cv

#Copy trained transformer from one pipeline to the other (ok)
mkdir -p /mnt/ml-team/dsb_2018/kamil/neptune_rescaled_patched/transformers
cp /mnt/ml-team/dsb_2018/kamil/neptune_size_estimator/transformers/unet_size_estimator \
/mnt/ml-team/dsb_2018/kamil/neptune_rescaled_patched/transformers/unet_size_estimator

# Fit the rescaled unet (ok)
neptune run --config configs_end_to_end/neptune_rescaled_patched.yaml \
-- train_pipeline --pipeline_name scale_adjusted_patched_unet_training --simple_cv

# Fit the missing transformers (those that are not trainable) (ok)
neptune run --config configs_end_to_end/neptune_rescaled_patched.yaml \
-- train_pipeline --pipeline_name scale_adjusted_patched_unet --simple_cv --dev_mode

# Evaluate pipeline
neptune run --config configs_end_to_end/neptune_rescaled_patched.yaml \
-- evaluate_pipeline --pipeline_name scale_adjusted_patched_unet --simple_cv

# Predict on test set in chunks
neptune run --config configs_end_to_end/neptune_rescaled_patched.yaml \
-- predict_pipeline --pipeline_name scale_adjusted_patched_unet --chunk_size 50
