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
 python3 training/trainers/train.py -cn=scf_whale \
 'mode=predict' \
 '~trainer.logger' \
 '+weights_path=/app/model_weights/scf/scf_whale.ckpt'
 #'+weights_path=/app/model_weights/scaleface/scaleface_whale.ckpt'
 #'+weights_path=/app/model_weights/pfe/pfe_whale.ckpt'
 #'+weights_path=/app/model_weights/scaleface/scaleface_vb2.ckpt'
 #'+weights_path=/app/model_weights/pfe/pfe_vb2.ckpt'
 #'+weights_path=/app/model_weights/scf/scf_vb2.ckpt'
 #'+weights_path=/app/model_weights/scf/scf_ms1m.ckpt' 
 #'+weights_path=/app/model_weights/scaleface/scaleface_ms1m.ckpt' 
 #'+weights_path=/app/model_weights/pfe/pfe_ms1m.ckpt' \ 

 # -cn=scaleface_whale -cn=pfe_whale -cn=scaleface_vb2 -cn=pfe_vb2 -cn=scf_vb2 -cn=scf_ms1m -cn=scaleface_ms1m # -cn=pfe_ms1m 