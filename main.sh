:'test model performance on VOCASET'

# test faceformer
python -u baseline_test.py --model faceformer_flame $* > logs/baseline_ff.log 2>&1 &

# test VOCA
python -u baseline_test.py --model voca $* > baseline_voca.log 2>&1 &

# test our model
python -u baseline_test.py --model tf_emo_4 $* > baseline_tf_emo_4.log 2>&1 &

:'model inference'