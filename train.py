from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from train_dataset import *
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from pytorch_lightning.callbacks import ModelCheckpoint

import os
os.environ['NCCL_P2P_DISABLE'] = '1'

# Configs
pre_train_path = "./models/v1-5-pruned.ckpt"
batch_size = 16
logger_freq = 5000 
learning_rate = 1e-5
sd_locked = True
only_mid_control = False

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/bps_sd15_pidinet.yaml').cpu()
model.load_state_dict(load_state_dict(pre_train_path, location='cpu'), strict=False)
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

dataset = COCODataset(source='pidinet') 
dataloader = DataLoader(dataset, num_workers=16, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
checkpoint_callback = ModelCheckpoint(
    #every_n_train_steps=2,
    every_n_epochs=5,
    save_top_k=-1
) 
trainer = pl.Trainer(gpus=[0],  
                     precision=16, 
                     accelerator="ddp",  
                     accumulate_grad_batches = 1, 
                     callbacks=[logger, checkpoint_callback],
                     default_root_dir = "./logs",
                     max_epochs = 16,
                     )


# Train!
if __name__ == '__main__':
    trainer.fit(model, dataloader)