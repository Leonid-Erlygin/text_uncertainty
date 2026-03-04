docker run \
 --shm-size=16g \
 --memory=160g \
 --cpus=40 \
 --user ${UID}:${UID} \
 --name ${USER}_$(basename $(dirname "$PWD"))_predict \
 --env HYDRA_FULL_ERROR=1 \
 --env MLFLOW_TRACKING_URI=$(cat /home/${USER}/face_ue/configs/mlflow_uri.yaml) \
 --rm \
 --init \
 -v $(dirname "$PWD"):/app \
 --gpus '"device=0"' \
 -w="/app" \
 ${USER}_$(basename $(dirname "$PWD")) \
 python3 osr_scripts/construct_protocol_topic_data.py

 
