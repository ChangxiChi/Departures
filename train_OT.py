from omegaconf import OmegaConf
from model import Model,train_dsbm,DSBMLightningModule
from Dataset.Preprocess import *
from Dataset.Datasets import *
import os
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
from datetime import datetime
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities.rank_zero import rank_zero_only
import torch.distributed as dist


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "conf", "sciplex3.yaml")
    # config_path = os.path.join(current_dir, "conf", "adamson.yaml")
    config = OmegaConf.load(config_path)
    print(config)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join(current_dir,"result", config['data_name'], timestamp)

    @rank_zero_only
    def prepare_dir(dir):
        os.makedirs(dir, exist_ok=True)

    prepare_dir(base_dir)

    logger = TensorBoardLogger(save_dir=base_dir, name="logs")

    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0

    # if rank==0:
    #     pert_data = PertData(
    #         hvg_num=config['gene_num'],
    #         pert_type=config['pert_type'],
    #         data_name=config['data_name'],
    #         threshold=config['threshold'],
    #         threshold_co=config['threshold_co']
    #     )
    #     broadcast_list = [pert_data]
    # else:
    #     broadcast_list = [None]

    # dist.broadcast_object_list(broadcast_list, src=0)

    # pert_data = broadcast_list[0]

    pert_data = PertData(
        hvg_num=config['gene_num'],
        pert_type=config['pert_type'],
        data_name=config['data_name'],
        threshold=config['threshold'],
        threshold_co=config['threshold_co']
    )

    data_module = MyDataModule_OT(
        pert_data=pert_data,
        pert_type=config['pert_type'],
        batch_size=config['batch_size']
    )
    data_module.setup()

    fwd_net_cfg = {
        "gene_num": config['gene_num'],
        "GRN": pert_data.GRN,
        "cell_type_num": pert_data.cell_type_num,
        "data_name": config['data_name'],
        "pert_type": config['pert_type'],
    }
    model = Model(num_steps=50, sig=0.2, **fwd_net_cfg)
    pl_model = DSBMLightningModule(model=model, lr=config['lr'], OT=True)

    trainer=Trainer(max_epochs=config['epochs'],
                    devices=-1,
                    accelerator="gpu",
                    strategy=DDPStrategy(find_unused_parameters=True),
                    logger=logger)

    trainer.fit(pl_model, datamodule=data_module)

    @rank_zero_only
    def save_model():
        ckpt_path = os.path.join(base_dir, "model_OT.ckpt")
        trainer.save_checkpoint(ckpt_path)
        print(f" Model checkpoint saved at: {ckpt_path}")

    save_model()



if __name__=="__main__":
    main()
