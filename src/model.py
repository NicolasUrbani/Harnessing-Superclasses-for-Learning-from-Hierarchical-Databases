import copy
import hydra
import lightning as L
import torch
import torch.nn.functional as F
import torchmetrics
from omegaconf import DictConfig

from .loss import HierarchicalLoss, HierarchicalRegularization
from .tree import add_weight_to_nodes
from .metrics import HierarchicalAccuracy, WassersteinTreeAccuracy, WassersteinTreeDistance, get_dist_matrix


class LightningModel(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.model = hydra.utils.instantiate(self.cfg.model.target)
        self.loss = hydra.utils.instantiate(self.cfg.loss.target)
        self.optimizer = self.cfg.optimizer
        self.lr_scheduler = self.cfg.lr_scheduler

        self.num_classes = self.cfg.model.out_features

        self.train_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.num_classes
        )
        self.val_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.num_classes
        )

        self.crm_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.num_classes
        )

        self.tree = hydra.utils.instantiate(self.cfg.dataset.datasets.hierarchy)
        add_weight_to_nodes(self.tree)
        self.wasserstein_acc = WassersteinTreeAccuracy(
            self.tree,
            average="macro"
        )

        self.wasserstein_dist = WassersteinTreeDistance(
            self.tree,
            average="macro"
        )

        self.hierarchical_accuracy = HierarchicalAccuracy(self.tree)
        self.crm_hier_acc = HierarchicalAccuracy(self.tree)
        self.wasserstein_tree_acc = WassersteinTreeAccuracy(self.tree)


        self.hierarchical_regularization = hydra.utils.instantiate(self.cfg.hierarchical_regularization.target)

        self.dist_matrix = get_dist_matrix(self.tree)

        self.hierarchical_regularization_gamma = hydra.utils.instantiate(self.cfg.hierarchical_regularization_gamma.target, dist_matrix=self.dist_matrix)



        if self.cfg.starting_point_regularization.param:
            w0 = []
            for name, weight in self.named_parameters_filtered(
                kind="weight", layer="butlast"
            ):
                w0.append(weight.detach().view(-1))
            self.register_buffer("w0", torch.cat(w0))

        hist_tree = copy.deepcopy(self.tree)
        add_weight_to_nodes(hist_tree, method="ones")
        self.dist_matrix = get_dist_matrix(hist_tree)

        
        if self.cfg.freeze_layers:
            for _, parameter in self.named_parameters_filtered(
                kind="weight", layer=self.cfg.freeze_layers
            ):
                parameter.require_grad = False
            for _, parameter in self.named_parameters_filtered(
                kind="bias", layer=self.cfg.freeze_layers
            ):
                parameter.require_grad = False
    def forward(self, x):
        return self.model.forward(x)

    def named_parameters_filtered(self, *, kind, layer):
        """Generate filtered name and parameter"""

        if kind not in ["weight", "bias"]:
            raise Exception("kind must be either `weight` or `bias`, found:", kind)

        if layer not in ["none", "last", "all", "butlast"]:
            raise Exception("kind must be either `none`, `last`, `all` or `butlast`, found:", layer)

        if layer == "none":
            return

        for name, parameter in self.model.named_parameters():
            if layer == "last" and name != self.model.last_layer_name:
                continue
            if layer == "butlast" and name == self.model.last_layer_name:
                continue
            if kind != "all" and kind in name:
                yield name, parameter

    def _shared_step(self, batch, batch_idx):
        input, target = batch
        output = self(input)

        preds = torch.argmax(output, dim=1)
        loss = self.loss(output, target)

        return output, preds, loss

    def training_step(self, batch, batch_idx):
        input, target = batch
        output, preds, loss = self._shared_step(batch, batch_idx)

        # Log train accuracy
        self.train_acc(preds, target)
        self.log("train_acc", self.train_acc)

        # Log train loss
        self.log("train_loss", loss)

        # Hierarchical regularization on last layer
        if self.cfg.hierarchical_regularization.param:
            weight = getattr(self.model, self.model.last_layer_name).weight
            hierarchical_regularization_loss = (
                self.cfg.hierarchical_regularization.param
                * self.hierarchical_regularization(weight)
            )
            self.log(
                "hierarchical_regularization_loss", hierarchical_regularization_loss
            )
        else:
            hierarchical_regularization_loss = 0.0

        if self.cfg.hierarchical_regularization_gamma.param:
            weight = getattr(self.model, self.model.last_layer_name).weight
            hierarchical_regularization_gamma_loss = (
                self.cfg.hierarchical_regularization_gamma.param
                * self.hierarchical_regularization_gamma(weight)
            )
            self.log(
                "hierarchical_regularization_gamma_loss", hierarchical_regularization_gamma_loss
            )
        else:
            hierarchical_regularization_gamma_loss = 0.0

        # Starting point regularization on all non-bias weights except for last
        # layer
        if self.cfg.starting_point_regularization.param:
            w = []
            for _, weight in self.named_parameters_filtered(
                kind="weight", layer="butlast"
            ):
                w.append(weight.view(-1))
            w = torch.cat(w)
            starting_point_regularization_loss = (
                self.cfg.starting_point_regularization.param
                * torch.linalg.norm(w - self.w0) ** 2
            )
            self.log("starting_point_loss", starting_point_regularization_loss)
        else:
            starting_point_regularization_loss = 0.0

        # Weight decay on remaining non-bias weights.
        if self.cfg.weight_decay.param:
            sp = self.cfg.starting_point_regularization.param
            hr = self.cfg.hierarchical_regularization.param

            if sp and hr:
                layer = "none"
            if sp and not hr:
                layer = "last"
            if not sp and hr:
                layer = "butlast"
            if not sp and not hr:
                layer = "all"

            weight_decay_loss = self.cfg.weight_decay.param * sum(
                torch.linalg.norm(p) ** 2
                for _, p in self.named_parameters_filtered(kind="weight", layer=layer)
            )
            self.log("weight_decay_loss", weight_decay_loss)
        else:
            weight_decay_loss = 0.0

        total_loss = (
            loss
            + hierarchical_regularization_loss
            + hierarchical_regularization_gamma_loss
            + weight_decay_loss
            + starting_point_regularization_loss
        )
        self.log("train_total_loss", total_loss)

        return total_loss

    def validation_step(self, batch, batch_idx):
        input, target = batch
        output, preds, loss = self._shared_step(batch, batch_idx)

        self.log("val_loss", loss, prog_bar=True)

        self.val_acc(preds, target)
        self.log("val_acc", self.val_acc)

        softmax_output = F.softmax(output, dim=1)
        self.wasserstein_dist(softmax_output, target)
        self.log("wasserstein_dist", self.wasserstein_dist)

        self.wasserstein_acc(output, target)
        self.log("wasserstein_acc", self.wasserstein_acc)

        self.hierarchical_accuracy(preds, target)
        self.log("hierarchical_accuracy", self.hierarchical_accuracy)

        crm = torch.argmin(torch.matmul(self.dist_matrix, softmax_output.T.cpu()), dim=0)
        self.crm_acc(crm, target.cpu())
        self.log("crm", self.crm_acc)
        self.crm_hier_acc(crm, target.cpu())
        self.log("crm_hier_acc", self.crm_hier_acc)

        self.wasserstein_tree_acc(output, target)
        self.log("wassersetin_tree_acc", self.wasserstein_tree_acc)

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self.optimizer.target, self.model.parameters())

        if self.lr_scheduler.target:
            lr_scheduler = hydra.utils.instantiate(self.lr_scheduler.target, optimizer)
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
        else:
            return {"optimizer": optimizer}

