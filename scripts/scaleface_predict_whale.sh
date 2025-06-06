docker run \
 --shm-size=16g \
 --memory=80g \
 --cpus=40 \
 --user ${UID}:${UID} \
 --name ${USER}_$(basename $(dirname "$PWD"))_predict \
 --env HYDRA_FULL_ERROR=1 \
 --rm \
 --init \
 -v $(dirname "$PWD"):/app \
 --gpus '"device=4"' \
 -w="/app" \
 ${USER}_$(basename $(dirname "$PWD")) \
 python3 training/trainers/train_multiple_runs_scf.py -cn=predict_scaleface_whale