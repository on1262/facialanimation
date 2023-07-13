# EXPRESSIVE SPEECH-DRIVEN FACIAL ANIMATION WITH CONTROLLABLE EMOTIONS

Source code for: [Expressive Speech-driven Facial Animation with controllable emotions](https://arxiv.org/abs/2301.02008)

**This repository is under construction now**

Update:

- initial update(Done, 6.18)
- rewrite framework (Done, 6.30)
- test inference module (Done, 6.30)
- test training module (Done, 7.1)
- check fitting algorithms
- rewrite schduler
- add deployment scripts

# Deployment

## Prepare dataset

If you want to train our model, you need to download CREMA-D, LRS2 and VOCASET dataset. Put dataset folders under `datasets` as following structure:

```
datasets
    CREMA-D
        AudioWAV
        VideoFLash
        ...
    LRS2
        mvlrs_v1
        train.txt
        val.txt
        ...
    VOCASET
        audio
        FaceTalk_XXXX_XXXX_TA
        ...
```

To get FLAME code dict for each sample, I use EMOCA to fit 2D datasets(CREMA-D and LRS2) and another fitting algorithm to fit 3D datasets(VOCASET). Uncomment the line under `make_dataset_cache` to fit 2D datasets(fitting algorithm for VOCASET is not available now). Results can be viewed in `datasets/cache` folder.

## Prepare 3rd packages

I use several 3rd packages for training and inference. All 3rd packages are under `third_party` folder. I provide a refactored version for `EMOCA`, which is called `EMOCABasic`. `EMOCABasic` rewrites the model interface, deleting unnecessary code for this model. But the checkpoint files and model weights are unchanged.

For training, `wav2vec2`, `DAN` and `EMOCABasic` are necessary. For inference(running model in inference mode), `wav2vec2` and `EMOCABasic` are needed. For baseline test, additional packages `Faceformer` and `VOCA` are needed.

File structure under `third_party` folder:

```
third_party
    DAN
    EMOCABasic
    Faceformer
    VOCA
    DAN
    wav2vec2
```

## Training

**training code is not available now**

- prepare datasets and 3rd packages following above instructions.
- run fitting algorithm for VOCASET
- run dataset caching scripts(uncomment the line under `make_dataset_cache` in `main.sh`)
- adjust `device` in `config/global.yml`
- adjust `train_minibatch` in `config/trainer.yml`, change `model_name` if you want to train another model.
- back to project folder, run `python -u main.py --mode train`

## Inference

**Read this before running inference**

I uploaded a `fusion` mode in inference scripts. It is only available in `aud-cls=XXX (HAP, DIS, FEA .etc)` mode (see inference configurations section) now. This method is able to disentangle lip movement and expression in output. It is actually a trick but I found it works well. Since lip movement and expression can be separately generated, the design of origin model is out of date. I will only keep inference mode available before I update the model. If you want to see the output of `fusion` mode, set sample configuration to `aud-cls=FEA` (or other labels), the rightmost is fusion output.

- download model weights and 3rd models from this link: https://1drv.ms/f/s!AomgXFHJMxuGk2jrH5UqrbYe5mTY?e=8VG12t
- put `date-Nov-/.../.pth` to tf_emo_4/saved_model
- `emoca_basic.ckpt` is the same as original EMOCA model weights (just name changed). Check MD5: 06a8d6bf2d9373ac2280a1bc7cf1acb4
- make sure that all files in `config/global.yml` are under correct paths
- input files for inference should be placed at `inference/input`
- change sample configs in `config/inference.yml/infer_dataset`, add custom input files and inference configs for each input.
- back to project folder, run `python -u inference.py --mode inference`
- get inference output at `inference/output`

# Implementation details

## Possibly occuring keys in sample dict

Collect Function:
- `wav`: audio data. tensor, shape: (batch, wav_len)
- `seqs_len`: video frame length for sequence, LongTensor, shape: (batch,)
- `params`: concat(expression code, pose code), tensor, shape: (batch, seq_len)
- `emo_logits`: DAN output, tensor, shape: (batch, 7)
- `code_dict`: dict contains FLAME code
    + `shapecode`: shape code. tensor, (batch, 100)
    + `expcode`: expression code. tensor, (batch, 50)
    + `posecode`: pose(head rotation, jaw angle), tensor, (batch, 6)
    + `lightcode`: code for light rendering, use `default_light_code`
    + `cam`: camera position, (batch, 3)
    + `texcode`: texture code for rendering, generated from EMOCA encoder.

EMOCABasic decoding results:
- `verts`: decoded mesh sequence by EMOCA based on `code_dict`, (batch, 5663)
- `output_images_coarse` and `predicted_images`: rendered images (grey or with texture)
- `geometry_coarse` and `emoca_imgs`: rendered grey face images without light
- `trans_verts`: temporal data for decoding process
- `masks`: `False` for background area in rendered images

Model input configurations:
- `name`: sample names, list
- `smooth`: enable output smoothing, bool
- `emo_label`: emotion control vector. Used to add emotion intensity in sequence level.
- `intensity`: custom intensity for emotion in `emo_label`
- `emo_logits_conf`: str, control model behavior:
    + `use`: predict emotion from autio without any change
    + `no_use`: generate model output without emotion
    + `one_hot`: adjust emotion based on `emo_label` and `intensity`

Datasets:
- `imgs`: original cropped image from 2D datasets
- `path`: fitting output path for VOCASET sample
- `wav_path`: wav path for VOCASET sample
- `flame_template`: template path for each sample in Baseline VOCASET
- `verts`: origin vertex data for each sample in Baseline VOCASET

## Inference configurations for each sample

`inference.yml/sample_configs` provides different configs to control model behavior. Some tags can be used independently, such as `video`, `audio`, `emo-ist`, `emo-cls`. Other tags should be used under specific situations, such as `-tex`, `=HAP`
- `video`: result order: [original video, EMOCA decoding output, `emo_logits_conf=use`, speech driven(predict emotion from audio), no emotion]
- `audio`: result order: [speech driven, no emotion]
- `emo-cls`: generate video with specific emotions: ['NEU','ANG','HAP','SAD','DIS','FEA']
- `aud-cls=XXX`: result order: [model output with emotion X enhancement, faceformer output, no emotion output]
- `emo-ist`: generate video with varying emotions and intensities, see `emo_ist` in `sample_configs`

# Experiments

**test loss in VOCASET**

Models  | Max loss(mm)| average loss(mm) |
--------- | --------| --------|
random init  | 3.87 | 2.31 |
faceformer | 3.33 | 1.97 |
voca | 3.41 | 1.94 |
tf_emo_4(mouth loss coeff=0.5) | 3.24 | 1.92 |
tf_emo_5(jaw loss coeff=0) | 3.22 | 1.92 |
tf_emo_8(few mask+LRS2) | 3.36 | 2.01
(conf A)tf_emo_2(mouth loss coeff=0) | 3.39 | 2.01 |
(conf B)tf_emo_6(add params mask) | 3.29 | 1.97 |
(conf C)tf_emo_3(reduce noise) | 3.32 | 1.99 |
(conf D)tf_emo_7(add params mask and introduces LRS2 dataset) | 3.34 | 1.99 |
(conf E)use only vocaset | 3.30 | 1.97 |
(conf F)tf_emo_10(disable transformer encoder) | 3.38 | 2.03 |