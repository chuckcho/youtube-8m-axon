#!/usr/bin/env bash

##############################################################################
# This script can be used to launch a job on batch ai within the cluster
# set up for youtube 8m. Please provide a model name and checkpoint number
##############################################################################

# Model name to run
model_name=$1

# checkpoint number
check_point=$2

export AZ_LOCATION=westus2
export AZ_RESOURCE_GROUP=BatchAI-sandbox

az configure --defaults location=$AZ_LOCATION
az configure --defaults group=$AZ_RESOURCE_GROUP

export CLUSTER_NAME=dus5uw2youtube8m-gpu
export CLUSTER_USER=clusteradmin
export WORKSPACE_NAME=Youtube8m
export EXPERIMENT_NAME=exp_run_yt8m_inference_20180727161944
export JOB_NAME=job_run_inference_${model_name}_$(date +"%Y%m%d%H%M%S")

job_spec_json_template='{
  "$schema": "https://raw.githubusercontent.com/Azure/BatchAI/master/schemas/2018-05-01/job.json",
  "properties": {
    "nodeCount": 1,
    "jobPreparation": {
      "commandLine": "apt update"
    },
    "customToolkitSettings": {
      "commandLine": "conda create --name batchai_py35 --clone py35; source activate batchai_py35; pip uninstall --yes tensorflow-gpu; conda install --yes tensorflow-gpu==1.8.0 ; cd $AZ_BATCHAI_MOUNT_ROOT/youtube8m/youtube-8m; sh axon-baseline-inference.sh MODEL CP"
    },
    "stdOutErrPathPrefix": "$AZ_BATCHAI_MOUNT_ROOT/youtube8m"
  }
}'

file_name=run_inference_for_$model_name.json

substitute_model=${job_spec_json_template[@]/MODEL/$model_name}
echo "Writing to $file_name"
echo "${substitute_model[@]/CP/$check_point}" > $file_name

az batchai job create --resource-group $AZ_RESOURCE_GROUP \
                      --name $JOB_NAME \
                      --cluster $CLUSTER_NAME \
                      --workspace $WORKSPACE_NAME \
                      --experiment $EXPERIMENT_NAME \
                      --config $file_name
