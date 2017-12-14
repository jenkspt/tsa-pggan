
TRAIN_FILES=gs://tsa_pggan_data/size_128/*.tfrecord
EVAL_FILES=gs://tsa_pggan_data/size_128/eval/*.tfrecord


export JOB_NAME=pggan_run5
export GCS_JOB_DIR=gs://tsa_pggan_data/logdir/$JOB_NAME
export CONFIG=config.yaml
export TRAIN_STEPS=100

gcloud ml-engine jobs submit training $JOB_NAME \
                                    --stream-logs \
                                    --runtime-version 1.4 \
				    --config $CONFIG \
                                    --job-dir $GCS_JOB_DIR \
                                    --module-name trainer.task \
                                    --package-path trainer/ \
                                    --region us-central1 \
                                    -- \
                                    --train-files $TRAIN_FILES \
                                    --eval-files $EVAL_FILES \
                                    --train-steps $TRAIN_STEPS \

#tensorboard --logdir=$GCS_JOB_DIR
