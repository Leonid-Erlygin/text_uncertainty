docker run \
 --shm-size=16g \
 --memory=80g \
 --cpus=40 \
 --user ${UID}:${UID} \
 --name ${USER}_$(basename $(dirname "$PWD"))_scf_train \
 --env WANDB_API_KEY=b2c5aadfb0bf526689d07a4bb4aae1eb58faf5b9 \
 --rm \
 --init \
 -v $(dirname "$PWD"):/app \
 --gpus '"device=0"' \
 -w="/app" \
 ${USER}_$(basename $(dirname "$PWD")) \
 python3 training/trainers/train_multiple_runs_scf.py -cn=train_scf_base