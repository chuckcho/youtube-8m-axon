# YouTube-8M Challenge (2018) for Axon
This repo has (1) all the latest developments from [official google starter code](https://github.com/google/youtube-8m), (2) [willow's rock solid baseline (especially gated NetVLAD model) and all their models](https://github.com/antoine77340/Youtube-8M-WILLOW) integrated, PLUS (3) axon-specific tools, scripts, and exploratory stuff. In other words, this *does* and *should* contains all the goodies for kaggle competition.
This README is Axon-specific documentation. See the [original google's README](README-google.md). Also, see the [willow's README](README-willow.md).

## Baseline training
Based on some quick experiments, last year's winning method appears a superb starting point. (1) Very fast convergence. (2) Very robust GAP result. (3) More stable training, hence, easy to reproduce. (I've seen training unstabilities using Google's official starter kit) Their code is from: https://github.com/antoine77340/Youtube-8M-WILLOW.

### Train frame-level NetVLAD with context-gate model as baseline
Run the following for the baseline model (of course, check data directory before running). Note that we use a half of the training data for faster training and ease of comparing with model changes. See: https://axon.quip.com/bOpyAw3mGmb3/YouTube-8M-Using-Subset-of-Validation-Set-to-Increase-Speed-of-Inference for details. See [axon-baseline-train.sh](axon-baseline-train.sh) for detail.
```
python train.py \
  --num_gpu=1 \
  --train_data_pattern="/media/6TB/videos/yt8m-v2/frame/train*[13579].tfrecord" \
  --train_dir=gatednetvladLF-256k-1024-80-0002-300iter-norelu-basic-gatedmoe \
  --model=NetVLADModelLF \
  --frame_features=True \
  --feature_names="rgb,audio" \
  --feature_sizes="1024,128" \
  --netvlad_cluster_size=256 \
  --netvlad_hidden_size=1024 \
  --moe_l2=1e-6 \
  --netvlad_relu=False \
  --gating=True \
  --moe_prob_gating=True \
  --base_learning_rate=0.0002 \
  --learning_rate_decay=0.8 \
  --max_step=500000 \
  --batch_size=64
```

### Validation performance
The GAP performance of the final model trained with the above command, evaluated on an Axon-official validate set should be approximatedly 85%. The following was used to run evaluation (note the validation is one tenth of the whole validation dataset). See [axon-baseline-eval.sh](axon-baseline-eval.sh) for detail.
```
python eval.py \
  --eval_data_pattern="/media/6TB/videos/yt8m-v2/frame/validate???5.tfrecord" \
  --train_dir=gatednetvladLF-256k-1024-80-0002-300iter-norelu-basic-gatedmoe \
  --netvlad_cluster_size=256 \
  --netvlad_hidden_size=1024 \
  --moe_l2=1e-6 \
  --netvlad_relu=False \
  --gating=True \
  --moe_prob_gating=True \
  --run_once=True \
  --batch_size=100 \
  --check_point=33209
```

### Inference for test data
If you submit the resulting inference CSV file to kaggle server, please document meticulously how you trained that model and/or how you ensembled multiple models (what models, weights for ensembling if not uniform).  Otherwise, we will easily lose track of all the good improvements we made, and won't be able to reproduce the results!
Add an entry to: https://axon.quip.com/nzbHAlae4bK7/Youtube-8M-Submission-to-Leaderboard.

The following will generate a csv file for kaggle submission (see [axon-baseline-inference.sh](axon-baseline-inference.sh) for detail):
```
python inference.py \
  --output_file=test_gatednetvladLF-256k-1024-80-0002-300iter-norelu-basic-gatedmoe.csv \
  --input_data_pattern=${axon_test_set} \
  --train_dir=gatednetvladLF-256k-1024-80-0002-300iter-norelu-basic-gatedmoe \
  --netvlad_cluster_size=256 \
  --netvlad_hidden_size=1024 \
  --moe_l2=1e-6 \
  --netvlad_relu=False \
  --gating=True \
  --moe_prob_gating=True \
  --top_k=50 \
  --batch_size=1024
```
`top_k` of 50 was used because in many cases inference results from multiple models will be blended and at the last stage, and only 20 best labels will be selected.

### Truncate top_k=40 into top_k=20 (good for submission)
The following command (see [truncate_20_labels.sh] for detail) will truncate top 20 labels from csv file generated with `top_k` > 20 (without having to run inference.py again with `top_k`=20):
```
cat test_gatednetvladLF-256k-1024-80-0002-300iter-norelu-basic-gatedmoe.csv \
  | awk -v n=41 'n==c{exit}n-c>=NF{print;c+=NF;next}{for(i=1;i<=n-c;i++)printf "%s ",$i;print x}' \
  > test_gatednetvladLF-256k-1024-80-0002-300iter-norelu-basic-gatedmoe_top20.csv
```
