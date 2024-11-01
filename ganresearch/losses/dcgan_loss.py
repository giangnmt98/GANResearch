import torch
import torch.nn as nn

from ganresearch.losses.losses import Losses


class DCGANLoss(Losses):
    def __init__(self, config):
        super().__init__(config)

    def calculate_loss(
        self, real_output, fake_output, is_discriminator=True, epoch=None
    ):
        loss_fn = nn.BCELoss()

        if is_discriminator:
            if self.ema is not None:
                # track the prediction
                self.ema.update(torch.mean(fake_output).item(), "D_fake", epoch)
                self.ema.update(torch.mean(real_output).item(), "D_real", epoch)
            # For discriminator: maximize log(D(x)) + log(1 - D(G(z)))
            real_labels = torch.ones_like(real_output)
            fake_labels = torch.zeros_like(fake_output)
            real_loss = loss_fn(real_output, real_labels)
            fake_loss = loss_fn(fake_output, fake_labels)
            if self.use_lecam and self.lecam_ratio > 0 and epoch > self.ema.start_epoch:
                loss_lecam = self.lecam_reg(real_loss, fake_loss) * self.lecam_ratio
            else:
                loss_lecam = torch.tensor(0.0)
            return real_loss + fake_loss + loss_lecam
        else:
            real_labels = torch.ones_like(
                fake_output
            )  # Target as real for generator loss
            return loss_fn(fake_output, real_labels)
