docker run \
 --shm-size=16g \
 --memory=80g \
 --cpus=40 \
 --user ${UID}:${UID} \
 --name ${USER}_$(basename $(dirname "$PWD"))_unc_train_3 \
 --env HYDRA_FULL_ERROR=1 \
 --env WANDB_API_KEY=$(cat /home/${USER}/face_ue/configs/wb_api.yaml) \
 --rm \
 --init \
 -v $(dirname "$PWD"):/app \
 --gpus '"device=0"' \
 -w="/app" \
 ${USER}_$(basename $(dirname "$PWD")) \
 python3 training/trainers/train.py -cn=arcface_cifar100N_r50
 #-cn=arcface_cifar10_clean # -cn=arcface_cifar10 #-cn=scf_whale # -cn=scaleface_whale #-cn=pfe_whale #-cn=scaleface_vb2 
 # -cn=pfe_vb2 # -cn=scf_vb2 #-cn=scf_ms1m # -cn=scaleface_ms1m # -cn=pfe_ms1m 