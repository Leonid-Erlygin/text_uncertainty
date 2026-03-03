docker run \
 --shm-size=16g \
 --memory=160g \
 --cpus=40 \
 --user ${UID}:${UID} \
 --name ${USER}_$(basename $(dirname "$PWD"))_arcface_train_v3 \
 --env HYDRA_FULL_ERROR=1 \
 --env MLFLOW_TRACKING_URI=$(cat /home/${USER}/face_ue/configs/mlflow_uri.yaml) \
 --rm \
 --init \
 -v $(dirname "$PWD"):/app \
 --gpus '"device=1"' \
 -w="/app" \
 ${USER}_$(basename $(dirname "$PWD")) \
 python3 training/trainers/train.py -cn=text_model_agnews #-cn=text_model_dbpedia #-cn=text_model_yahoo #-cn=text_model_blog #-cn=text_model_pan # # #-cn=text_model_clinc150 #-cn=text_model