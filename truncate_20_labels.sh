#!/usr/bin/env bash

# it's assumed that IN file has (label_id, label_score) pairs that are sorted in
# a decreasing order.

# specify input/output files
IN=
OUT=

cat $IN \
  | awk -v n=41  \
  'n==c{exit}n-c>=NF{print;c+=NF;next}{for(i=1;i<=n-c;i++)printf "%s ",$i;print x}'  \
  > $OUT
