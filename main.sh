# test model performance on VOCASET

# inference
# python -u inference.py --mode inference

# train
# python -u main.py --mode train

# scripts

# make dataset cache
python -u main.py --mode scripts --scripts dataset_cache > logs/scripts_dataset_cache.log 2>&1 &