docker run \
 --shm-size=16g \
 --memory=80g \
 --cpus=40 \
 --user ${UID}:${UID} \
 --name ${USER}_$(basename $(dirname "$PWD"))_vb2_embs \
 --env HYDRA_FULL_ERROR=1 \
 --rm \
 -it \
 --init \
 -v $(dirname "$PWD"):/app \
 --gpus '"device=4"' \
 -w="/app/sandbox/scripts" \
 ${USER}_$(basename $(dirname "$PWD")) \
 bash