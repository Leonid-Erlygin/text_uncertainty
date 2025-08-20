docker run \
 --shm-size=16g \
 --memory=160g \
 --cpus=40 \
 --user ${UID}:${UID} \
 --name ${USER}_$(basename $(dirname "$PWD"))_unc_train \
 --env HYDRA_FULL_ERROR=1 \
 --env MLFLOW_TRACKING_URI=$(cat /home/${USER}/face_ue/configs/mlflow_uri.yaml) \
 --rm \
 --init \
 -v $(dirname "$PWD"):/app \
 --gpus '"device=5"' \
 -w="/app" \
 ${USER}_$(basename $(dirname "$PWD")) \
 python3 training/trainers/train.py -cn=due_scf #-cn=scf_cifar10_noisy #-cn=scf_cifar10_clean #-cn=scf_cifar10_noisy #-cn=arcface_cifar10N_32x32_clean #-cn=arcface_cifar10N_32x32 #