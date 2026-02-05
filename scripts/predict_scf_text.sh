docker run \
 --shm-size=16g \
 --memory=160g \
 --cpus=40 \
 --user ${UID}:${UID} \
 --name ${USER}_$(basename $(dirname "$PWD"))_unc_train_2 \
 --env HYDRA_FULL_ERROR=1 \
 --env MLFLOW_TRACKING_URI=$(cat /home/${USER}/face_ue/configs/mlflow_uri.yaml) \
 --rm \
 --init \
 -v $(dirname "$PWD"):/app \
 --gpus '"device=5"' \
 -w="/app" \
 ${USER}_$(basename $(dirname "$PWD")) \
 python3 training/trainers/train.py -cn=text_model_blog_scf \
 'mode=predict' \
 '~trainer.logger' \
 '+weights_path="/app/outputs/text_scf/blog_4k/epoch=7-step=3880.ckpt"'

#  python3 training/trainers/train.py -cn=text_model_pan_scf \
#  'mode=predict' \
#  '~trainer.logger' \
#  '+weights_path="/app/outputs/text_scf/pan_4k/epoch=7-step=1296.ckpt"'

#  python3 training/trainers/train.py -cn=text_model_clinc150_scf \
#  'mode=predict' \
#  '~trainer.logger' \
#  '+weights_path="/app/outputs/text_scf/clinc150/epoch=3-step=468.ckpt"'