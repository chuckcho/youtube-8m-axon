#!/usr/bin/env bash

wget --no-clobber http://us.data.yt8m.org/2/ground_truth_labels/train_labels.csv
wget --no-clobber http://us.data.yt8m.org/2/ground_truth_labels/validate_labels.csv

# merge all labels
cat train_labels.csv validate_labels.csv | sort --field-separator=',' --key=1 > all_labels.csv
