import torch
import torch.nn as nn
import numpy as np
import importlib
from matplotlib import cm, ticker
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier


class BoostingCalibration:
    def __init__(self, log_dir):
        self.log_dir = log_dir

    def train_calibration_parameters(self, kl_1, kl_2, true_pred_label, save_name):
        X = np.concatenate([kl_1[None, :], kl_2[None, :]], axis=0).T
        y = true_pred_label
        self.X_mean_val = np.mean(X, axis=0)
        self.X_std_val = np.std(X, axis=0)
        X_norm = (X - self.X_mean_val) / self.X_std_val
        self.clf = GradientBoostingClassifier(
            n_estimators=500,
            learning_rate=1.0,
            max_depth=1,
            random_state=0,
            validation_fraction=0.05,
        ).fit(X_norm, y)

        self.draw_dencity_plot(X_norm, y, save_name)

    def apply_calibration_transform(self, kl_1, kl_2, y, save_name):
        X = np.concatenate([kl_1[None, :], kl_2[None, :]], axis=0).T
        self.X_mean_test = np.mean(X, axis=0)
        self.X_std_test = np.std(X, axis=0)
        X_norm = (X - self.X_mean_test) / self.X_std_test
        self.draw_dencity_plot(X_norm, y, save_name)
        print(f"Score: {self.clf.score(X_norm, y)}")
        predictions_boosting = self.clf.predict_proba(X_norm)[:, 1]
        unc = -predictions_boosting
        return unc

    def draw_dencity_plot(self, X_norm, y, image_name):
        size = 500
        kl_1 = torch.linspace(X_norm[:, 0].min(), X_norm[:, 0].max(), size)
        kl_2 = torch.linspace(X_norm[:, 1].min(), X_norm[:, 1].max(), size)
        grid_x, grid_y = np.meshgrid(
            kl_1.cpu().numpy(), kl_2.cpu().numpy(), indexing="ij"
        )
        product = torch.cartesian_prod(kl_1, kl_2)

        predict_prob = self.clf.predict_proba(product.detach().cpu().numpy())[:, 1]
        z = np.reshape(predict_prob, (size, size)).T
        z_min, z_max = z.min(), z.max()

        fig, ax = plt.subplots()
        cs = ax.contourf(grid_x, grid_y, z, cmap=cm.PuBu_r, vmin=z_min, vmax=z_max)
        ax.axis([grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()])
        cbar = fig.colorbar(cs)
        sns.scatterplot(
            data={
                "kl_1": X_norm[:, 0],
                "kl_2": X_norm[:, 1],
                "true_pred_label": y,
            },
            x="kl_1",
            y="kl_2",
            hue="true_pred_label",
            s=1,
            alpha=0.5,
        )
        log_dir = Path(self.log_dir) / "calibration_images_boosting"
        log_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(log_dir / f"{image_name}.png")


class IDResBlock(nn.Module):
    def __init__(self, input_size, hidden_size: int, use_bn=True):
        super().__init__()
        self.bn0 = nn.BatchNorm1d(input_size, affine=True)

        self.mlp1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size, affine=True)
        self.activ = nn.Sigmoid()

        self.mlp2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size, affine=True)

        self.mlp3 = nn.Linear(hidden_size, hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size, affine=True)

        self.use_bn = use_bn

    def forward(self, x):
        if self.use_bn:
            x = self.bn0(x)
        x = self.mlp1(x)
        if self.use_bn:
            x = self.bn1(x)
        x = self.activ(x)
        identity = x
        x = self.mlp2(x)
        if self.use_bn:
            x = self.bn2(x)
        x = self.activ(x)
        x = self.mlp3(x)
        if self.use_bn:
            x = self.bn3(x)
        out = x + identity
        return out


class NNcalibration:
    def __init__(
        self,
        hidden_size,
        num_layers,
        use_bn,
        lr,
        epochs,
        weight,
        weight_decay,
        scheduler_params,
        train_weight=True,
        normalize_kl_by_test=False,
        random_subset_size=None,
        log_dir=None,
    ):
        self.device = torch.device("cuda")
        num_layers = num_layers
        base_dim = hidden_size
        layers = []
        prev_dim = 2
        use_bn = use_bn
        for i in range(num_layers):
            new_dim = base_dim * (i + 2) * 4
            layers.extend(
                [
                    IDResBlock(prev_dim, new_dim, use_bn),
                ]
            )
            prev_dim = new_dim

        if use_bn:
            layers.append(nn.BatchNorm1d(prev_dim, affine=True))
        layers.extend(
            [
                nn.Linear(prev_dim, 1),
                nn.Sigmoid(),
                nn.Flatten(start_dim=0),
            ]
        )
        self.perceptron = nn.Sequential(*layers)
        self.perceptron.to(self.device)
        self.lr = lr
        self.epochs = epochs
        self.weight = weight
        self.weight_decay = weight_decay
        self.scheduler_params = scheduler_params
        self.random_subset_size = random_subset_size
        self.log_dir = log_dir
        self.normalize_kl_by_test = normalize_kl_by_test
        self.train_weight = train_weight

    def train_calibration_parameters(self, kl_1, kl_2, error_calc, save_name):
        X = torch.tensor(
            np.concatenate([kl_1[None, :], kl_2[None, :]], axis=0).T,
            dtype=torch.float32,
            device=self.device,
        )
        true_pred_label = np.zeros(error_calc.is_seen.shape[0])
        true_pred_label[error_calc.is_seen] = error_calc.true_accept_true_ident
        true_pred_label[~error_calc.is_seen] = error_calc.true_reject
        # save validation normalization parameters
        self.X_mean_val = torch.mean(X, dim=0)
        self.X_std_val = torch.std(X, dim=0)
        X_norm = (X - self.X_mean_val) / self.X_std_val
        y = torch.tensor(
            true_pred_label.astype("bool"), dtype=torch.float32, device=self.device
        )

        if self.weight is None:
            true_pred_ratio = y.sum() / y.shape[0]
            print(true_pred_ratio.item())
            self.weight = true_pred_ratio.item()
        # weight = torch.tensor(self.weight, device=self.device)
        if self.train_weight:
            weight = torch.nn.Parameter(
                torch.tensor(self.weight, device=self.device), requires_grad=True
            )
        else:
            weight = torch.tensor(self.weight, device=self.device)

        scheduler_params = {
            "scheduler": "OneCycleLR",
            "params": {
                "max_lr": self.scheduler_params.max_lr,
                "steps_per_epoch": self.scheduler_params.steps_per_epoch,
                "epochs": self.epochs,
                "div_factor": self.scheduler_params.div_factor,
                "final_div_factor": self.scheduler_params.final_div_factor,
            },
            "interval": "epoch",
            "frequency": 1,
        }

        # loss_fn = nn.BCELoss(weight=weights)
        loss_fn = nn.BCELoss(reduce=False)
        self.perceptron.train()

        if self.random_subset_size is not None:
            weights = torch.zeros(
                int(X_norm.shape[0] * self.random_subset_size), device=self.device
            )
        else:
            pass

        if self.train_weight:
            optimizer = torch.optim.Adam(
                [*self.perceptron.parameters()] + [weight],
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        else:
            optimizer = torch.optim.Adam(
                [*self.perceptron.parameters()],
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        scheduler = getattr(
            importlib.import_module("torch.optim.lr_scheduler"),
            scheduler_params["scheduler"],
        )(optimizer, **scheduler_params["params"])

        true_index = y == 1.0
        for iter in range(self.epochs):
            self.perceptron.train()
            optimizer.zero_grad()
            # sample train ds subset
            if self.random_subset_size is not None:
                indices = torch.randperm(X_norm.shape[0])[
                    : int(X_norm.shape[0] * self.random_subset_size)
                ]
                X_norm_subset = X_norm[indices]
                y_subset = y[indices]
                weights[y_subset == 1.0] = 1 - weight
                weights[y_subset == 0.0] = weight
                pred = self.perceptron(X_norm_subset)
                loss = (loss_fn(pred, y_subset) * weights).mean()
            else:
                pred = self.perceptron(X_norm)
                loss_element_wise = loss_fn(pred, y)
                loss = loss_element_wise[true_index].mean() * (
                    1 - torch.sigmoid(weight)
                ) + loss_element_wise[~true_index].mean() * torch.sigmoid(weight)
                # loss = (loss_fn(pred, y)* weights).mean()

            loss.backward()
            optimizer.step()
            scheduler.step()
            # print(
            #     f"Iteration {iter}, Loss: {loss.item()}, lr: {optimizer.param_groups[0]['lr']}"
            # )

            self.perceptron.eval()
            pred_eval = self.perceptron(X_norm)
            accuracy = np.mean(
                (pred_eval.detach().cpu().numpy() > 0.5) == y.cpu().numpy()
            )
            print(
                f"Iteration {iter}, Loss: {loss.item()}, accuracy: {accuracy.item()}, lr: {optimizer.param_groups[0]['lr']}"
            )
            print(torch.sigmoid(weight).item())
        # draw probs
        self.draw_dencity_plot(X_norm.cpu(), error_calc, save_name)

    def apply_calibration_transform(self, kl_1, kl_2, error_calc, save_name):

        X = torch.tensor(
            np.concatenate([kl_1[None, :], kl_2[None, :]], axis=0).T,
            dtype=torch.float32,
            device=self.device,
        )
        if self.normalize_kl_by_test:
            self.X_mean_test = torch.mean(X, dim=0)
            self.X_std_test = torch.std(X, dim=0)
            X_norm = (X - self.X_mean_test) / self.X_std_test
        else:
            X_norm = (X - self.X_mean_val) / self.X_std_val
        self.perceptron.eval()
        predictions_perceptron = self.perceptron(X_norm)
        self.draw_dencity_plot(X_norm.cpu(), error_calc, save_name)
        unc = -predictions_perceptron.detach().cpu().numpy()
        return unc

    def draw_dencity_plot(self, X_norm, error_calc, image_name):
        true_pred = np.zeros(error_calc.is_seen.shape[0])
        true_pred[error_calc.is_seen] = error_calc.true_accept_true_ident
        true_pred[~error_calc.is_seen] = error_calc.true_reject
        false_accept = np.zeros(error_calc.is_seen.shape[0])
        false_accept[~error_calc.is_seen] = error_calc.false_accept
        false_ident_or_false_reject = np.zeros(error_calc.is_seen.shape[0])
        false_ident_or_false_reject[error_calc.is_seen] = (
            ~error_calc.true_accept_true_ident
        )

        size = 500
        kl_1 = torch.linspace(
            X_norm[:, 0].min(), X_norm[:, 0].max(), size, device=self.device
        )
        kl_2 = torch.linspace(
            X_norm[:, 1].min(), X_norm[:, 1].max(), size, device=self.device
        )
        grid_x, grid_y = np.meshgrid(
            kl_1.cpu().numpy(), kl_2.cpu().numpy(), indexing="ij"
        )
        product = torch.cartesian_prod(kl_1, kl_2)

        self.perceptron.eval()
        predict_prob = self.perceptron(product)
        z = np.reshape(predict_prob.detach().cpu().numpy(), (size, size)).T
        z_min, z_max = z.min(), z.max()

        fig, ax = plt.subplots()
        cs = ax.contourf(grid_x, grid_y, z, cmap=cm.PuBu_r, vmin=z_min, vmax=z_max)
        ax.axis([grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()])
        cbar = fig.colorbar(cs)
        pred_kind = []
        for i in range(error_calc.is_seen.shape[0]):
            if true_pred[i]:
                pred_kind.append("no error")
            elif false_accept[i]:
                pred_kind.append("false accept")
            elif false_ident_or_false_reject[i]:
                pred_kind.append("false ident or reject")
        assert len(pred_kind) == error_calc.is_seen.shape[0]
        hue_order = ["no error", "false accept", "false ident or reject"]
        kl_data = pd.DataFrame(
            list(zip(X_norm[:, 0].numpy(), X_norm[:, 1].numpy(), pred_kind)),
            columns=["kl_1", "kl_2", "prediction kind"],
        )
        sns.scatterplot(
            data=kl_data.sort_values(
                "prediction kind", key=np.vectorize(hue_order.index)
            ),
            x="kl_1",
            y="kl_2",
            hue="prediction kind",
            hue_order=hue_order,
            s=10,
            alpha=0.5,
        )
        log_dir = Path(self.log_dir) / "calibration_images"
        log_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(log_dir / f"{image_name}.png")


class Standartization:
    def __init__(self, normalize_kl_by_test=False, log_dir=None, vis=False):
        self.normalize_kl_by_test = normalize_kl_by_test
        self.log_dir = log_dir
        self.vis = vis
        self.device = torch.device("cpu")

    def train_calibration_parameters(self, kl_1, kl_2, error_calc, save_name):
        X = torch.tensor(
            np.concatenate([kl_1[None, :], kl_2[None, :]], axis=0).T,
            dtype=torch.float32,
            device=self.device,
        )
        true_pred_label = np.zeros(error_calc.is_seen.shape[0])
        true_pred_label[error_calc.is_seen] = error_calc.true_accept_true_ident
        true_pred_label[~error_calc.is_seen] = error_calc.true_reject
        # save validation normalization parameters
        self.X_mean_val = torch.mean(X, dim=0)
        self.X_std_val = torch.std(X, dim=0)
        X_norm = (X - self.X_mean_val) / self.X_std_val

        # draw probs
        if self.vis:
            self.draw_dencity_plot(X_norm.cpu(), error_calc, save_name)

    def apply_calibration_transform(self, kl_1, kl_2, error_calc, save_name):

        X = torch.tensor(
            np.concatenate([kl_1[None, :], kl_2[None, :]], axis=0).T,
            dtype=torch.float32,
            device=self.device,
        )
        if self.normalize_kl_by_test:
            self.X_mean_test = torch.mean(X, dim=0)
            self.X_std_test = torch.std(X, dim=0)
            X_norm = (X - self.X_mean_test) / self.X_std_test
        else:
            X_norm = (X - self.X_mean_val) / self.X_std_val
        if self.vis:
            self.draw_dencity_plot(X_norm.cpu(), error_calc, save_name)
        kl_sum = torch.sum(X_norm, dim=1)  # self.perceptron(X_norm)
        unc = -kl_sum.detach().cpu().numpy()
        return unc

    def draw_dencity_plot(self, X_norm, error_calc, image_name):
        true_pred = np.zeros(error_calc.is_seen.shape[0])
        true_pred[error_calc.is_seen] = error_calc.true_accept_true_ident
        true_pred[~error_calc.is_seen] = error_calc.true_reject
        false_accept = np.zeros(error_calc.is_seen.shape[0])
        false_accept[~error_calc.is_seen] = error_calc.false_accept
        false_ident_or_false_reject = np.zeros(error_calc.is_seen.shape[0])
        false_ident_or_false_reject[error_calc.is_seen] = (
            ~error_calc.true_accept_true_ident
        )

        size = 500
        kl_1 = torch.linspace(
            X_norm[:, 0].min(), X_norm[:, 0].max(), size, device=self.device
        )
        kl_2 = torch.linspace(
            X_norm[:, 1].min(), X_norm[:, 1].max(), size, device=self.device
        )
        grid_x, grid_y = np.meshgrid(
            kl_1.cpu().numpy(), kl_2.cpu().numpy(), indexing="ij"
        )
        product = torch.cartesian_prod(kl_1, kl_2)

        predict_prob = torch.sum(product, dim=1)
        z = np.reshape(predict_prob.detach().cpu().numpy(), (size, size)).T
        z_min, z_max = z.min(), z.max()

        fig, ax = plt.subplots()
        cs = ax.contourf(grid_x, grid_y, z, cmap=cm.PuBu_r, vmin=z_min, vmax=z_max)
        ax.axis([grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()])
        cbar = fig.colorbar(cs)
        pred_kind = []
        for i in range(error_calc.is_seen.shape[0]):
            if true_pred[i]:
                pred_kind.append("no error")
            elif false_accept[i]:
                pred_kind.append("false accept")
            elif false_ident_or_false_reject[i]:
                pred_kind.append("false ident or reject")
        assert len(pred_kind) == error_calc.is_seen.shape[0]
        hue_order = ["no error", "false accept", "false ident or reject"]
        kl_data = pd.DataFrame(
            list(zip(X_norm[:, 0].numpy(), X_norm[:, 1].numpy(), pred_kind)),
            columns=["kl_1", "kl_2", "prediction kind"],
        )
        sns.scatterplot(
            data=kl_data.sort_values(
                "prediction kind", key=np.vectorize(hue_order.index)
            ),
            x="kl_1",
            y="kl_2",
            hue="prediction kind",
            hue_order=hue_order,
            s=10,
            alpha=0.5,
        )
        log_dir = Path(self.log_dir) / "calibration_images"
        log_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(log_dir / f"{image_name}.png")
