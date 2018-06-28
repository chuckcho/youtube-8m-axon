# YouTube-8M Challenge (2018) for Axon
This README is Axon-specific documentation. See the [original google's README](README-google.md).
This repo contains all the codes and scripts for youtube-8m-2018 kaggle challenge.

## Baseline training
Since we know that frame-level features yield better performance (than video-level ones), the following frame-level model should be our "baseline".

### Frame-level LSTM model as baseline
Run the following for the baseline model. Note that we use a half of the training data for faster training and ease of comparing with model changes.
```
python train.py
  --frame_features \
  --model=LstmModel  \
  --feature_names='rgb,audio' \
  --feature_sizes='1024,128' \
  --train_data_pattern=${HOME}/yt8m/v2/frame/train???[13579].tfrecord \
  --train_dir ./sample_model \
  --batch_size=256 \
  --start_new_model
```
Batch size of 256 works on a GPU with 12GB memory. [`train-baseline-its-getting-dark.sh`](train-baseline-its-getting-dark.sh) run this command, intended to work on `dextro-its-getting-dark` machine.

### Baseline performance (GAP)
The GAP performance of the final model, evaluated on an Axon-official validate set should be approximately XX.XX% (TODO). The following was used to run evaluation (note the validation is one tenth of the whole validation dataset).
```
python eval.py
  --eval_data_pattern=${HOME}/yt8m/v2/frame/validate???5.tfrecord \
  --train_dir ./sample_model
```
