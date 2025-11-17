# code is built on “https://github.com/yuyang-shi/dsbm-pytorch”
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm
from functools import partial
import copy

from typing import List, Optional, Tuple
import pytorch_lightning as pl
from module import model,model_discrete
import os
from datetime import datetime
import pytorch_lightning as pl

# adamson sota
# no x_0 input ver

device="cuda"
# Departures
class Model(nn.Module):
    def __init__(self, num_steps=1000, sig=0, eps=1e-3, first_coupling="random", **kwargs):
        super().__init__()
        self.net_fwd = model(**kwargs)
        self.net_mask = model_discrete(**kwargs)
        self.N = num_steps
        self.sig = sig
        self.eps = eps
        self.first_coupling = first_coupling
  
    @torch.no_grad()
    def get_train_tuple(self, x_pairs=None, prediction_type="z1"):
        z0, z1 = x_pairs[:, 0], x_pairs[:, 1]
        t = torch.rand((z1.shape[0], 1), device=device) * (1-2*self.eps) + self.eps
        z_t = t * z1 + (1.-t) * z0
        z = torch.randn_like(z_t)
        z_t = z_t + self.sig * torch.sqrt(t*(1.-t)) * z

        if prediction_type=="v":
        # z1 - z_t / (1-t)
            target = z1 - z0 
            target = target - self.sig * torch.sqrt(t/(1.-t)) * z
        else:
            # z1 prediction
            target = z1
        
        return z_t, t, target
    
    @torch.no_grad()
    def get_train_tuple_discrete(self, x_d_pairs=None):
        z0, z1 = x_d_pairs[:, 0], x_d_pairs[:, 1]
        batch_size, dim = z0.shape
        t = torch.rand((batch_size,), device=device) * (1 - 2 * self.eps) + self.eps
        t = t.view(-1, 1)
        mask = torch.rand((batch_size, dim), device=device) < t

        # zt = phi(t) * z0 + (1-phi(t)) * z1
        z_t = torch.where(mask, z1, z0)
        target = z1 

        return z_t, t, target


    @torch.no_grad()
    def sample_sde(self, zstart=None, N=None, knockout=None, cell_type=None, mole=None, dosage=None):
        ### NOTE: Use Euler method to sample from the learned flow
        if N is None:
            N = self.N   
        dt = 1./N
        # traj = [] # to store the trajectory
        z = zstart.detach().clone()
        z0 = zstart.detach().clone()
        batchsize = z.shape[0]
        
        # traj.append(z.detach().clone())
        ts = np.arange(N) / N

        for i in range(N):
            t = torch.ones((batchsize,1), device=device) * ts[i]
            pred = self.net_fwd(z,
                                t,
                                z0,
                                knockout=knockout,
                                cell_type=cell_type,
                                mole=mole,
                                dosage=dosage)
            v = (pred-z)/(1-t)
            z = z.detach().clone() + v * dt 
            z = z + self.sig * torch.randn_like(z) * np.sqrt(dt)
            # traj.append(z.detach().clone())

        return z
        
    # old version
    # @torch.no_grad()
    # def sample_discrete(self, zstart=None, N=None, knockout=None, cell_type=None, mole=None, dosage=None, threshold=None):
    #     """
    #     Discrete sampling of mask model.
    #     """
    #     if N is None:
    #         N = self.N
    #     h = 1.0 / N

    #     z = zstart.detach().clone()  # [B, D], long tensor of token indices
    #     # traj = [z.clone()]
    #     t = 0.0

    #     for i in range(N):
    #         t_tensor = torch.ones_like(z[:, :1], device=z.device) * t  # [B, 1]

    #         logits = self.net_mask(z, t_tensor,
    #                                knockout=knockout,
    #                                 cell_type=cell_type,
    #                                 mole=mole,
    #                                 dosage=dosage)  # [B, D, V]
    #         probs = torch.softmax(logits, dim=-1)  # [B, D, V]
    #         # One-hot of current state
    #         one_hot_z = nn.functional.one_hot(z.long(), num_classes=probs.size(-1)).float()  # [B, D, V]
    #         u = (probs - one_hot_z) / (1.0 - t + 1e-5)  # velocity field
    #         new_probs = one_hot_z + h * u  # forward Euler
    #         new_probs = torch.clamp(new_probs, min=1e-8, max=1.0)  # avoid degenerate probs
    #         new_probs = new_probs / new_probs.sum(dim=-1, keepdim=True)  # renormalize
    #         # Sample from the updated probs
    #         if threshold is None:
    #             z = torch.distributions.Categorical(probs=new_probs).sample()  # [B, D]
    #             # traj.append(z.clone())
    #         else:
    #             z = (new_probs[:, :, 1]>threshold).long()

    #         t += h

    #     return z


    @torch.no_grad()
    def sample_discrete(self, zstart=None, z0=None, N=None, knockout=None, cell_type=None, mole=None, dosage=None, threshold=None):
        """
        Discrete sampling of mask model.
        """
        if N is None:
            N = self.N
        h = 1.0 / N

        z = zstart.detach().clone()  # [B, D], long tensor of token indices
        z0 = z0.detach().clone()
        # traj = [z.clone()]
        t = 0.0

        for i in range(N):
            t_tensor = torch.ones_like(z[:, :1], device=z.device) * t  # [B, 1]

            logits = self.net_mask(z, 
                                   t_tensor,
                                   z0,
                                   knockout=knockout,
                                    cell_type=cell_type,
                                    mole=mole,
                                    dosage=dosage)  # [B, D]
            probs = torch.sigmoid(logits).unsqueeze(-1)  # [B, D, 1]
            probs = torch.cat([(1.0 - probs).to(zstart.device), probs], dim=-1) # [B, D, 2]
            # One-hot of current state
            one_hot_z = nn.functional.one_hot(z.long(), num_classes=probs.size(-1)).float()  # [B, D, 2]
            u = (probs - one_hot_z) / (1.0 - t + 1e-5)  # velocity field
            new_probs = one_hot_z + h * u  # forward Euler
            new_probs = torch.clamp(new_probs, min=1e-8, max=1.0)  # avoid degenerate probs
            new_probs = new_probs / new_probs.sum(dim=-1, keepdim=True)  # renormalize
            # Sample from the updated probs
            if threshold is None:
                z = torch.distributions.Categorical(probs=new_probs).sample()  # [B, D]
                # traj.append(z.clone())
            else:
                z = (new_probs[:, :, 1]>threshold).long()

            t += h

        return z


def train_dsbm(dsbm, dataloader, epochs=40, lr=1e-3, checkpoint_save_path=""):
    optimizer = torch.optim.Adam(dsbm.net_fwd.parameters(), lr=lr)
    optimizer_d = torch.optim.Adam(dsbm.net_mask.parameters(), lr=lr)
    loss_curve = []
    loss_curve_d = []

    for epoch in range(epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            # expression prediciton model
            x_pairs = torch.stack([batch['x0'], batch['x1']], dim=1)
            x_t, t, target = dsbm.get_train_tuple(x_pairs)
            optimizer.zero_grad()
            pred = dsbm.net_fwd(x_t, 
                                t,
                                batch['x0'],
                                knockout=batch.get('knockout',None), 
                                cell_type=batch['cell_type'], 
                                mole=batch.get('mole',None), 
                                dosage=batch.get('dosage',None))
            loss = (batch['x1_d'].detach()*(target - pred)).pow(2).sum()
            loss = loss/(batch['x1_d'].detach().sum())
            loss.backward()

            # gene slience prediciton model
            x_d_pairs = torch.stack([batch['x0_d'], batch['x1_d']], dim=1)
            x_d_t, t, target_d = dsbm.get_train_tuple_discrete(x_d_pairs)
            optimizer_d.zero_grad()
            pred_d = dsbm.net_mask(x_d_t, 
                                    t,
                                    batch['x0'],
                                    knockout=batch.get('knockout',None), 
                                    cell_type=batch['cell_type'], 
                                    mole=batch.get('mole',None), 
                                    dosage=batch.get('dosage',None))
            
            # with torch.no_grad():
            #     label_counts = torch.bincount(target.flatten(), minlength=2).float()
            #     class_weights = 1.0 / (label_counts + 1e-6)
            #     class_weights = class_weights / class_weights.sum()

            loss_d = nn.functional.cross_entropy(
                                    pred_d.flatten(0, 1),
                                    target_d.flatten(0, 1).long(),
                                    # weight=class_weights.to(pred_d.device)
                                )
            loss_d.backward()   

        
            if torch.isnan(loss).any():
                raise ValueError("Loss is nan")
                break
            
            optimizer.step()
            optimizer_d.step()
            loss_curve.append(loss.item())
            loss_curve_d.append(loss_d.item())

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'loss_d': f'{loss_d.item():.4f}'
            })
    
    file_path = os.path.abspath(__file__)
    dir_path = os.path.dirname(file_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    full_dir = os.path.join(dir_path, checkpoint_save_path, timestamp)
    os.makedirs(full_dir, exist_ok=True)
    filename = "model.pt"
    full_path = os.path.join(full_dir, filename)

    torch.save(dsbm.state_dict(), full_path)

    loss_curve_np = np.array(loss_curve)
    loss_curve_d_np = np.array(loss_curve_d)
    np.save(os.path.join(full_dir, "loss_curve.npy"), loss_curve_np)
    np.save(os.path.join(full_dir, "loss_curve_d.npy"), loss_curve_d_np)

    return loss_curve, loss_curve_d


class DSBMLightningModule(pl.LightningModule):
    def __init__(self, model: torch.nn.Module, lr=1e-3, OT=False):
        super().__init__()
        self.model = model
        self.lr = lr
        self.OT = OT

    def training_step(self, batch, batch_idx):
        # forward net_fwd part
        x_pairs = torch.stack([batch['x0'], batch['x1']], dim=1)
        x_t, t, target = self.model.get_train_tuple(x_pairs)
        pred = self.model.net_fwd(
            x_t, 
            t,
            batch['x0'],
            knockout=batch.get('knockout', None),
            cell_type=batch['cell_type'],
            mole=batch.get('mole', None),
            dosage=batch.get('dosage', None)
        )
        loss_fwd = (batch['x1_d'].detach() * (target - pred)).pow(2).sum()
        loss_fwd = loss_fwd / (batch['x1_d'].detach().sum())

        # forward net_mask part
        x_d_pairs = torch.stack([batch['x0_d'], batch['x1_d']], dim=1)
        x_d_t, t, target_d = self.model.get_train_tuple_discrete(x_d_pairs)
        pred_d = self.model.net_mask(
                                    x_d_t, 
                                    t,
                                    batch['x0'],
                                    knockout=batch.get('knockout', None),
                                    cell_type=batch['cell_type'],
                                    mole=batch.get('mole', None),
                                    dosage=batch.get('dosage', None)
                                )

        # with torch.no_grad():
        #     label_counts = torch.bincount(target_d.flatten().long(), minlength=2).float()
        #     class_weights = 1.0 / (label_counts + 1e-6)
        #     class_weights = class_weights / class_weights.sum()

        # loss_mask = F.cross_entropy(
        #                             pred_d.flatten(0, 1),
        #                             target_d.flatten(0, 1).long(),
        #                             weight=class_weights.to(pred_d.device)
        #                         )
        logits = torch.sigmoid(pred_d)
        loss_mask = F.binary_cross_entropy(logits, target_d.float())

        self.log('loss_fwd', loss_fwd, prog_bar=True, on_step=True, on_epoch=True)
        self.log('loss_mask', loss_mask, prog_bar=True, on_step=True, on_epoch=True)

        total_loss = loss_fwd + loss_mask
        self.log('loss_total', total_loss, prog_bar=True, on_step=True, on_epoch=True)

        if torch.isnan(total_loss):
            raise ValueError("Loss is nan!")

        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        return optimizer
    
    @torch.no_grad()
    def predict(self, ctrl_expression, N=None, knockout=None, cell_type=None, mole=None, dosage=None, threshold=None):
        self.model.eval()
        pred=self.model.sample_sde(zstart=ctrl_expression, 
                                   N=N, 
                                   knockout=knockout, 
                                   cell_type=cell_type, 
                                   mole=mole, 
                                   dosage=dosage)
        
        ctrl_d = (ctrl_expression > 0).float()
        pred_d=self.model.sample_discrete(zstart=ctrl_d,
                                        z0=ctrl_expression,
                                        N=N,
                                        knockout=knockout,
                                        cell_type=cell_type,
                                        mole=mole,
                                        dosage=dosage,
                                        threshold=threshold)
        
        Prediciton=pred*pred_d
        return Prediciton,pred,pred_d


    def on_train_epoch_start(self):
        if self.OT:
            self.trainer.datamodule.train_dataset.resample_OT()
        else:
            self.trainer.datamodule.train_dataset.resample()         
