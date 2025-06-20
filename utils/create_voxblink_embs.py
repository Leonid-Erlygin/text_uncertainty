import torch
from sandbox.ScriptsForVoxBlink2.asv.modules.model_spk import ResNet34_based
from sandbox.ScriptsForVoxBlink2.asv.modules.feat import logFbankCal
from sandbox.ScriptsForVoxBlink2.asv.dataset import WavDataset
from training.dataset_classes.lightning_datasets import VoxBlinkDataset
from torchsummary import summary
import numpy as np
import argparse
from torch import multiprocessing
from tqdm import tqdm

torch.multiprocessing.set_start_method("spawn")
NUM_IDS_TO_SAVE = -1


def get_model():
    featCal = logFbankCal(
        sample_rate=16000, n_fft=512, win_length=0.025, hop_length=0.01, n_mels=80
    )
    model = ResNet34_based(
        in_planes=64,
        block_type="base",
        pooling_layer="GSP",
        embd_dim=256,
        acoustic_dim=80,
        featCal=featCal,
    )
    state_dict = torch.load(
        "/app/sandbox/ScriptsForVoxBlink2/ckpt/resnet34/model_ft.pt", map_location="cpu"
    )
    model.load_state_dict(
        {k.replace("module.", ""): v for k, v in state_dict.items()}, strict=False
    )
    model.to("cuda:0")
    model.eval()
    return model


def predict_embeddings(process_id, num_splits):
    full_ds = VoxBlinkDataset(
        "/app/datasets/VB1"
    )  # VoxBlinkDataset("/app/datasets/VB2_11/SMIIPdata1/AudioData/VoxBlink2")
    # full_ds = torch.utils.data.Subset(full_ds, np.arange(800))
    max_id = len(full_ds)  # 2786740
    splits = np.split(np.arange(max_id), num_splits)
    ds = torch.utils.data.Subset(full_ds, splits[process_id])
    embs = []
    bottlenecks = []
    model = get_model()
    with torch.no_grad():
        for i in tqdm(range(len(ds))):
            signal = ds[i][0][None, ...].to("cuda:0")
            emb, bottleneck = model(signal)
            embs.append(emb.cpu().numpy())
            bottlenecks.append(bottleneck.cpu().numpy())
    embs = np.concatenate(embs, axis=0)
    bottlenecks = np.concatenate(bottlenecks, axis=0)
    np.savez(f"outputs/embs_vb_{process_id}.npz", embs=embs, bottlenecks=bottlenecks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("process_id")
    parser.add_argument("num_splits")
    args = parser.parse_args()

    predict_embeddings(int(args.process_id), int(args.num_splits))
