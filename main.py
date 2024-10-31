# import argparse
#
# from ganresearch.training.trainer import Trainer
#
# from ganresearch.dataloader.dataloader import DataLoaderManager
# # from evaluation.fid_score import FIDScore
# from ganresearch.models.gan_factory import GANFactory
# from ganresearch.utils.utils import load_config
#
#
# def main(config):
#
#     # Chuẩn bị DataLoader
#     dataloader_manager = DataLoaderManager(config)
#     train_loader, val_loader, _ = dataloader_manager.get_dataloaders()
#     # Khởi tạo mô hình
#     gan_factory = GANFactory(config)
#     gan_model = gan_factory.create_model_gan()
#
#     # Khởi tạo Trainer với mô hình và DataLoader
#     trainer = Trainer(gan_model, config, train_loader, val_loader)
#
#     # Bắt đầu huấn luyện với Early Stopping và lưu biểu đồ loss
#     trainer.train()
#
#     # # Đánh giá bằng FID Score
#     # fid = FIDScore(real_images=[], generated_images=[])
#     # fid_score = fid.get_fid_scores()
#     # print(f"FID Score: {fid_score}")
#
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="GAN Training and Evaluation")
#     parser.add_argument(
#         "--config", type=str, default="config/config.yaml", help="Path to config file"
#     )
#     args = parser.parse_args()
#
#     # Đọc file config
#     config = load_config(args.config)
#     # Chạy chương trình chính
#     main(config)


import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
from torch.utils.data import DataLoader, random_split, Subset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision import  models
import numpy as np
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
import torch.nn.functional as F


class Config:
    def __init__(self):
        self.dataset = 'cifar10'
        self.dataroot = 'data'
        self.workers = 2
        self.batch_size = 64
        self.image_size = 64
        self.nz = 100
        self.ngf = 64
        self.ndf = 64
        self.niter = 1
        self.lr = 0.0002
        self.beta1 = 0.5
        self.lecam_ratio = 0.1
        self.use_lecam = False
        self.cuda = torch.cuda.is_available()
        self.ngpu = 1
        self.outf = 'result'
        self.manual_seed = random.randint(1, 10000) if not self.cuda else None

opt = Config()

def initialize():
    os.makedirs(opt.outf, exist_ok=True)
    if opt.manual_seed:
        print("Random Seed: ", opt.manual_seed)
        random.seed(opt.manual_seed)
        torch.manual_seed(opt.manual_seed)
    if opt.cuda:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    return device

class ema_losses:
    def __init__(self, init=1000., decay=0.9, start_itr=0):
        self.G_loss = init
        self.D_loss_real = init
        self.D_loss_fake = init
        self.D_real = init
        self.D_fake = init
        self.decay = decay
        self.start_itr = start_itr

    def update(self, cur, mode, itr):
        if itr < self.start_itr:
            decay = 0.0
        else:
            decay = self.decay
        if mode == 'G_loss':
          self.G_loss = self.G_loss*decay + cur*(1 - decay)
        elif mode == 'D_loss_real':
          self.D_loss_real = self.D_loss_real*decay + cur*(1 - decay)
        elif mode == 'D_loss_fake':
          self.D_loss_fake = self.D_loss_fake*decay + cur*(1 - decay)
        elif mode == 'D_real':
          self.D_real = self.D_real*decay + cur*(1 - decay)
        elif mode == 'D_fake':
          self.D_fake = self.D_fake*decay + cur*(1 - decay)


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d(opt.nz, opt.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(opt.ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(opt.ngf * 8, opt.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(opt.ngf * 4, opt.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(opt.ngf * 2, opt.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(opt.ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(3, opt.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(opt.ndf, opt.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(opt.ndf * 2, opt.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(opt.ndf * 4, opt.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(opt.ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output.view(-1, 1).squeeze(1)

def load_data():
    transform = transforms.Compose([
        transforms.Resize(opt.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    if opt.dataset == 'cifar10':
        dataset = dset.CIFAR10(root=opt.dataroot, download=True, transform=transform)
    else:
        raise ValueError("Unsupported dataset")
    return torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers)

def lecam_reg(dis_real, dis_fake, ema):
    import torch.nn.functional as F
    reg = torch.mean(F.relu(dis_real - ema.D_fake).pow(2)) + \
            torch.mean(F.relu(ema.D_real - dis_fake).pow(2))
    return reg


def create_loss(use_lecam=False):
    """
    Computes the loss for the DCGAN model based on whether it's used for the
    discriminator or generator.

    Args:
        real_output (torch.Tensor): The discriminator output for real images.
        fake_output (torch.Tensor): The discriminator output for fake images generated by the generator.
        is_discriminator (bool): If True, computes loss for the discriminator.
                                 If False, computes loss for the generator.

    Returns:
        torch.Tensor: The computed loss.
    """
    # Use Binary Cross-Entropy Loss
    def loss(real_output, fake_output, is_discriminator=True, ema=None, it=None):
        loss_fn = nn.BCELoss()
        if is_discriminator:
            if ema is not None:
                # track the prediction
                ema.update(torch.mean(fake_output).item(), 'D_fake', it)
                ema.update(torch.mean(real_output).item(), 'D_real', it)
            # For discriminator: maximize log(D(x)) + log(1 - D(G(z)))
            real_labels = torch.ones_like(real_output)
            fake_labels = torch.zeros_like(fake_output)
            real_loss = loss_fn(real_output, real_labels)
            fake_loss = loss_fn(fake_output, fake_labels)
            if use_lecam and opt.lecam_ratio > 0 and it > ema.start_itr:
                loss_lecam = lecam_reg(real_loss, fake_loss, ema) * opt.lecam_ratio
            else:
                loss_lecam = torch.tensor(0.)
            return real_loss + fake_loss + loss_lecam
        else:
            # For generator: maximize log(D(G(z)))
            real_labels = torch.ones_like(fake_output)  # Target as real for generator loss
            return loss_fn(fake_output, real_labels)
    return loss


def train_step(epoch, dataloader, disciminator, generator , optimizer_d, optimizer_g, loss_function, ema_losses, device, g_loss_total):
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        disciminator.zero_grad()
        real_cpu = data[0].to(device)
        batch_size = real_cpu.size(0)

        # Train with real
        real_output = disciminator(real_cpu)
        D_x = real_output.mean().item()
        # Generate fake images
        noise = torch.randn(batch_size, opt.nz, 1, 1, device=device)
        fake = generator(noise)
        fake_output = disciminator(fake.detach())
        d_loss = loss_function(real_output, fake_output, is_discriminator=True, ema=ema_losses, it=i)
        d_loss.backward()
        d_g_z1 = fake_output.mean().item()
        optimizer_d.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        generator.zero_grad()
        fake_output = disciminator(fake)
        g_loss = loss_function(None, fake_output, is_discriminator=False)
        g_loss.backward()
        d_g_z2 = fake_output.mean().item()
        g_loss_total += g_loss.item()
        optimizer_g.step()

        # Accumulation
        ema_losses.update(g_loss_total, 'G_loss', i)
        # Logging (optional)
        if i % 100 == 0:
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch, opt.niter, i, len(dataloader),
                     d_loss.item(), g_loss.item(), D_x, d_g_z1, d_g_z2))
    return {
        "real_cpu": real_cpu,
        "generator": generator,
        "d_loss": d_loss.item(),
        "g_loss": g_loss.item(),
        "D_x": D_x,
        "D_G_z1": d_g_z1,
        "D_G_z2": d_g_z2
    }

def train(dataloader, disciminator, generator , optimizer_d, optimizer_g, loss_function, ema_losses, device):
    # Initialize lists to store loss values
    d_losses = []
    g_losses = []
    fixed_noise = torch.randn(opt.batch_size, opt.nz, 1, 1, device=device)
    for epoch in range(opt.niter):
        g_loss_total = 0
        result_dict = train_step(epoch, dataloader, disciminator, generator , optimizer_d, optimizer_g, loss_function, ema_losses, device, g_loss_total)
        real_cpu, generator, d_loss, g_loss, D_x, D_G_z1, D_G_z2 = result_dict.values()
        d_losses.append(d_loss)
        g_losses.append(g_loss)
        vutils.save_image(real_cpu,
                          '%s/real_samples.png' % opt.outf,
                          normalize=True)
        fake = generator(fixed_noise)
        vutils.save_image(fake.detach(),
                          '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
                          normalize=True)

        # do checkpointing
        # torch.save(generator.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
        # torch.save(disciminator.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))
    return  d_losses, g_losses

def plot_losses(d_losses, g_losses, save_path='loss_plot.png'):
    plt.figure(figsize=(12, 6))
    plt.plot(d_losses, label="Discriminator Loss", color='red', linewidth=2)
    plt.plot(g_losses, label="Generator Loss", color='blue', linewidth=2)

    # Thêm tiêu đề và các nhãn phụ
    plt.title("Generator and Discriminator Losses Over Iterations", fontsize=16)
    plt.xlabel("Iterations", fontsize=14)
    plt.ylabel("Loss", fontsize=14)

    # Thêm lưới (grid) cho biểu đồ
    plt.grid(True)

    # Cài đặt hiển thị cho chú thích (legend)
    plt.legend(fontsize=12)

    # Thêm chú giải vào các vị trí cụ thể (Optional)
    # để người xem dễ dàng biết được giá trị mất mát tại một số điểm cụ thể
    for i in range(0, len(d_losses), len(d_losses) // 5):  # điểm mẫu
        plt.text(i, d_losses[i], f"{d_losses[i]:.2f}", fontsize=10, ha='center')
        plt.text(i, g_losses[i], f"{g_losses[i]:.2f}", fontsize=10, ha='center')

    # Lưu biểu đồ vào đường dẫn
    plt.savefig(save_path)

    # # Hiển thị biểu đồ
    # plt.show()


# Create datasets
def create_datasets(imbalance_ratios, batch_size=64):
  train_transform = transforms.Compose([
      transforms.Resize(opt.image_size),
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
  ])

  fid_transform = transforms.Compose([
      transforms.Resize(299),
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
  ])

  cifar10_train = dset.CIFAR10(root="./data", train=True, download=True, transform=train_transform)
  cifar10_test = dset.CIFAR10(root="./data", train=False, download=True, transform=fid_transform)

  train_size = int(0.8 * len(cifar10_train))
  val_size = len(cifar10_train) - train_size
  train_dataset, val_dataset = random_split(cifar10_train, [train_size, val_size])

  targets = np.array([cifar10_train.targets[i] for i in train_dataset.indices])
  indices = [i for class_id, ratio in imbalance_ratios.items()
             for i in np.where(targets == class_id)[0][:int(len(np.where(targets == class_id)[0]) * ratio)]]

  imbalanced_dataset = Subset(train_dataset, indices)

  return (
      DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True),
      DataLoader(imbalanced_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True),
      DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True),
      DataLoader(cifar10_test, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
  )

# Calculate FID score between two sets of features
def calculate_fid(real_features, fake_features):
  mu_real, sigma_real = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
  mu_fake, sigma_fake = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)

  covmean = sqrtm(sigma_real @ sigma_fake)
  if np.iscomplexobj(covmean):
      covmean = covmean.real

  fid = np.sum((mu_real - mu_fake) ** 2) + np.trace(sigma_real + sigma_fake - 2 * covmean)
  return fid

# Extract features class-by-class
def extract_class_features(loader, model, class_id, generator=None):
  model.eval()
  features = []
  with torch.no_grad():
      for inputs, labels in loader:
          mask = labels == class_id  # Filter inputs by class
          if mask.sum() == 0:
              continue
          inputs = inputs[mask].to(device)

          if generator:
              # Generate fake images using the generator
              noise = torch.randn(inputs.size(0), opt.nz, 1, 1, device=device)
              inputs = generator(noise)

          # Resize inputs to 299x299 for Inception-v3
          inputs = F.interpolate(inputs, size=(299, 299), mode='bilinear', align_corners=False)
          outputs = model(inputs)  # Extract features
          features.append(outputs.cpu().numpy())

  return np.concatenate(features, axis=0)

# Compare FID scores per class
def compare_fid_scores(test_loader, generator_balanced, generator_imbalanced, imbalance_ratios):
  # Initialize the Inception-v3 model for feature extraction
  from torchvision.models import Inception_V3_Weights

  inception = models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False).to(device)
  inception.fc = torch.nn.Identity()  # Replace the FC layer

  fid_data = []

  for class_id in range(10):  # Assuming CIFAR-10 has 10 classes
      print(f"Processing class {class_id}...")

      # Extract real features for the current class
      real_features = extract_class_features(test_loader, inception, class_id)

      # Extract fake features from the balanced and imbalanced generators
      fake_features_balanced = extract_class_features(test_loader, inception, class_id, generator_balanced)
      fake_features_imbalanced = extract_class_features(test_loader, inception, class_id, generator_imbalanced)

      # Calculate FID scores
      fid_balanced = calculate_fid(real_features, fake_features_balanced)
      fid_imbalanced = calculate_fid(real_features, fake_features_imbalanced)

      # Store results for this class
      fid_data.append({
          "Class": class_id,
          "FID (Balanced)": fid_balanced,
          "FID (Imbalanced)": fid_imbalanced,
          "FID (Imbalanced) - FID (Balanced)": fid_imbalanced - fid_balanced
      })

  # Display the results
  import pandas as pd
  df = pd.DataFrame(fid_data)
  df["Imbalance Ratio"] = df["Class"].map(imbalance_ratios)
  print(df)

  return df

def run_train(data_loader, ema_losses, device):
    discriminator = Discriminator(opt.ngpu).to(device)
    generator = Generator(opt.ngpu).to(device)

    # setup optimizer
    optimizerD = optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    loss = create_loss(use_lecam=opt.use_lecam)
    d_losses, g_losses = train(dataloader=data_loader, disciminator=discriminator,
                               generator=generator, optimizer_d=optimizerD,
                               optimizer_g=optimizerG, loss_function=loss, ema_losses=ema_losses, device=device)
    # plot_losses(d_losses, g_losses)
    return generator

if __name__ == "__main__":
    # Initialize and train the model

    # Imbalance ratios for each class
    imbalance_ratios = {0: 0.01, 1: 0.01, 2: 0.02, 3: 0.05, 4: 0.4, 5: 0.5, 6: 0.6, 7: 0.7, 8: 0.8, 9: 0.9}
    # Create datasets
    train_loader_balanced, train_loader_imbalanced, val_loader, test_loader = create_datasets(imbalance_ratios)

    device = initialize()
    ema_losses = ema_losses(start_itr=100)

    real_label = 1
    fake_label = 0

    generator_balanced = run_train(train_loader_balanced, ema_losses, device)

    generator_imbalanced = run_train(train_loader_imbalanced, ema_losses, device)

    compare_fid_scores(test_loader, generator_balanced, generator_imbalanced, imbalance_ratios)