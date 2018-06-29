# YouTube-8M Challenge (2018) for Axon
This README is Axon-specific documentation. See the [original google's README](README-google.md).
This repo contains all the codes and scripts for youtube-8m-2018 kaggle challenge.

## Baseline training
Based on some quick experiments, last year's winning method appears a superb starting point. (1) Very fast convergence. (2) Very robust GAP result. (3) More stable training, hence, easy to reproduce. (I've seen training unstabilities using Google's official starter kit) Their code is from: https://github.com/antoine77340/Youtube-8M-WILLOW. Slight modifications were made to address data format changes, and their code is in [willow directory](willow).

### Frame-level NetVLAD with context-gate model as baseline
Run the following for the baseline model (of course, check data directory before running). Note that we use a half of the training data for faster training and ease of comparing with model changes. See: https://axon.quip.com/bOpyAw3mGmb3/YouTube-8M-Using-Subset-of-Validation-Set-to-Increase-Speed-of-Inference for details.
```
cd willow
python train.py \
  --train_data_pattern="/media/6TB/videos/yt8m-v2/frame/train*[13579].tfrecord" \
  --model=NetVLADModelLF \
  --train_dir=gatednetvladLF-256k-1024-80-0002-300iter-norelu-basic-gatedmoe \
  --frame_features=True \
  --feature_names="rgb,audio" \
  --feature_sizes="1024,128" \
  --batch_size=80 \
  --base_learning_rate=0.0002 \
  --netvlad_cluster_size=256 \
  --netvlad_hidden_size=1024 \
  --moe_l2=1e-6 \
  --iterations=300 \
  --learning_rate_decay=0.8 \
  --netvlad_relu=False \
  --gating=True \
  --moe_prob_gating=True \
  --max_step=500000
```

### Baseline performance (GAP)
The GAP performance of the final model, evaluated on an Axon-official validate set should be 84.7%~85.5%. The following was used to run evaluation (note the validation is one tenth of the whole validation dataset).
```
cd willow
python eval.py \
  --eval_data_pattern="/media/6TB/videos/yt8m-v2/frame/validate???5.tfrecord" \
  --model=NetVLADModelLF \
  --train_dir=gatednetvladLF-256k-1024-80-0002-300iter-norelu-basic-gatedmoe \
  --frame_features=True \
  --feature_names="rgb,audio" \
  --feature_sizes="1024,128" \
  --batch_size=200 \
  --base_learning_rate=0.0002 \
  --netvlad_cluster_size=256 \
  --netvlad_hidden_size=1024 \
  --moe_l2=1e-6 \
  --iterations=300 \
  --learning_rate_decay=0.8 \
  --netvlad_relu=False \
  --gating=True \
  --moe_prob_gating=True \
  --run_once=True \
  --top_k=50 \
  --check_point=64140
```
For three different trainings (yet identical params/settings), GAP values of *84.73%, 85.48%, 85.44%* were obtained.

### Inference for test data
Likewise, the following will generate a csv file for kaggle submission:
```
cd willow
python inference.py \
  --output_file=test_gatednetvladLF-256k-1024-80-0002-300iter-norelu-basic-gatedmoe.csv \
  --input_data_pattern="/media/6TB/videos/yt8m-v2/frame/test????.tfrecord" \
  --model=NetVLADModelLF \
  --train_dir=gatednetvladLF-256k-1024-80-0002-300iter-norelu-basic-gatedmoe \
  --frame_features=True \
  --feature_names="rgb,audio" \
  --feature_sizes="1024,128" \
  --batch_size=1024 \
  --base_learning_rate=0.0002 \
  --netvlad_cluster_size=256 \
  --netvlad_hidden_size=1024 \
  --moe_l2=1e-6 \
  --iterations=300 \
  --learning_rate_decay=0.8 \
  --netvlad_relu=False \
  --gating=True \
  --moe_prob_gating=True \
  --run_once=True \
  --top_k=50 \
  --check_point=64140
```
`top_k` of 50 was used because in many cases inference results from multiple models will be blended and at the last stage, and only 20 best labels will be selected.
The following command will truncate top 20 labels from csv file generated with `top_k` > 20 (without having to run inference.py again with `top_k`=20):
`cat $IN | awk -v n=41 'n==c{exit}n-c>=NF{print;c+=NF;next}{for(i=1;i<=n-c;i++)printf "%s ",$i;print x}' > $OUT`
