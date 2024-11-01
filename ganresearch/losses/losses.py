import torch
import torch.nn.functional as F


class Losses:
    def __init__(
        self,
        config,
    ):
        self.use_lecam = bool(config["training"]["use_lecam"])
        self.lecam_ratio = float(config["training"]["lecam_ratio"])
        if self.use_lecam and self.lecam_ratio > 0:
            self.ema = EMA(
                int(config["training"]["init_ema"]),
                float(config["training"]["decay_ema"]),
                int(config["training"]["start_epoch_ema"]),
            )
        else:
            self.ema = None

    def lecam_reg(self, real_loss, fake_loss):
        reg = torch.mean(F.relu(real_loss - self.ema.D_fake).pow(2)) + torch.mean(
            F.relu(self.ema.D_real - fake_loss).pow(2)
        )
        return reg

    def calculate_loss(self, is_discriminator=True, epoch=None):
        pass


class EMA:
    def __init__(self, init=100, decay=0.9, start_epoch=0):
        self.G_loss = init
        self.D_loss_real = init
        self.D_loss_fake = init
        self.D_real = init
        self.D_fake = init
        self.decay = decay
        self.start_epoch = start_epoch

    def update(self, cur, mode, itr):
        if itr < self.start_epoch:
            decay = 0.0
        else:
            decay = self.decay
        if mode == "G_loss":
            self.G_loss = self.G_loss * decay + cur * (1 - decay)
        elif mode == "D_loss_real":
            self.D_loss_real = self.D_loss_real * decay + cur * (1 - decay)
        elif mode == "D_loss_fake":
            self.D_loss_fake = self.D_loss_fake * decay + cur * (1 - decay)
        elif mode == "D_real":
            self.D_real = self.D_real * decay + cur * (1 - decay)
        elif mode == "D_fake":
            self.D_fake = self.D_fake * decay + cur * (1 - decay)
