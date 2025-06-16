docker run \
 --shm-size=16g \
 --memory=80g \
 --cpus=40 \
 --user ${UID}:${UID} \
 --name ${USER}_$(basename $(dirname "$PWD"))_unc_predict \
 --env HYDRA_FULL_ERROR=1 \
 --rm \
 --init \
 -v $(dirname "$PWD"):/app \
 --gpus '"device=5"' \
 -w="/app" \
 ${USER}_$(basename $(dirname "$PWD")) \
 python3 training/trainers/train.py -cn=scf \
 'mode=predict' \
 '~trainer.logger' \
 '+weights_path=/app/model_weights/scf_base.ckpt' #'+weights_path=/app/model_weights/scale_face.ckpt' #'+weights_path=/app/model_weights/pfe_base.ckpt' \ 

 # -cn=scaleface # -cn=pfe 