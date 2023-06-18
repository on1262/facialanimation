cd /home/chenyutong/facialanimation
python -u inference.py \
--vid_path "/home/chenyutong/facialanimation/Visualize/infer_source_video" \
--cache_path "/home/chenyutong/facialanimation/Visualize/cache" \
--infer_path "/home/chenyutong/facialanimation/Visualize/infer_sample" \
--gt_path "/home/chenyutong/facialanimation/Visualize/gt_sample" \
$*