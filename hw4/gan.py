import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Callable
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer


class Discriminator(nn.Module):
    def __init__(self, in_size):
        """
        :param in_size: The size of on input image (without batch dimension).
        """
        super().__init__()
        self.in_size = in_size
        # TODO: Create the discriminator model layers.
        #  To extract image features you can use the EncoderCNN from the VAE
        #  section or implement something new.
        #  You can then use either an affine layer or another conv layer to
        #  flatten the features.
        # ====== YOUR CODE: ======
        modules = [
            nn.Conv2d(in_size[0], 64, kernel_size=5),
            nn.ReLU(),
            nn.Conv2d(64, 256, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm2d(512),
            nn.Dropout2d(p=0.1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=5, padding=6, stride=2),
            nn.BatchNorm2d(512),
            nn.Dropout2d(p=0.1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm2d(512),
            nn.Dropout2d(p=0.1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=5, padding=7, stride=2),
            nn.BatchNorm2d(512),
            nn.Dropout2d(p=0.1),
            nn.ReLU(),
            nn.Conv2d(512, 1, kernel_size=8),
            nn.ReLU(),
        ]
        self.cnn = nn.Sequential(*modules)

    # ========================

    def forward(self, x):
        """
        :param x: Input of shape (N,C,H,W) matching the given in_size.
        :return: Discriminator class score (not probability) of
        shape (N,).
        """
        # TODO: Implement discriminator forward pass.
        #  No need to apply sigmoid to obtain probability - we'll combine it
        #  with the loss due to improved numerical stability.
        # ====== YOUR CODE: ======
        X_shape = x.shape[0]
        y = self.cnn(x)
        y_newshape = y.reshape((X_shape, 1))
        # ========================
        return y_newshape


class Generator(nn.Module):
    def __init__(self, z_dim, featuremap_size=4, out_channels=3):
        """
        :param z_dim: Dimension of latent space.
        :featuremap_size: Spatial size of first feature map to create
        (determines output size). For example set to 4 for a 4x4 feature map.
        :out_channels: Number of channels in the generated image.
        """
        super().__init__()
        self.z_dim = z_dim
        self.featuremap_size = featuremap_size

        # TODO: Create the generator model layers.
        #  To combine image features you can use the DecoderCNN from the VAE
        #  section or implement something new.
        #  You can assume a fixed image size.
        # ====== YOUR CODE: ======
        modules = [
            nn.Upsample(scale_factor=2, mode="bicubic"),

            nn.Conv2d(int(z_dim / (self.featuremap_size ** 2)), 256, kernel_size=5, padding=2),
            nn.Dropout2d(0.1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=5, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bicubic"),
            nn.Conv2d(256, 128, kernel_size=5, padding=2, dilation=1, stride=1),
            nn.Dropout2d(0.1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=5, padding=2, dilation=1, stride=1),
            nn.Dropout2d(0.1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bicubic"),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bicubic"),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv2d(64, out_channels, kernel_size=5, padding=2, dilation=1),
            nn.ReLU(),
        ]
        self.cnn = nn.Sequential(*modules)
        # ========================

    def sample(self, n, with_grad=False):
        """
        Samples from the Generator.
        :param n: Number of instance-space samples to generate.
        :param with_grad: Whether the returned samples should be part of the
        generator's computation graph or standalone tensors (i.e. should be
        be able to backprop into them and compute their gradients).
        :return: A batch of samples, shape (N,C,H,W).
        """
        device = next(self.parameters()).device
        # TODO: Sample from the model.
        #  Generate n latent space samples and return their reconstructions.
        #  Don't use a loop.
        # ====== YOUR CODE: ======
        sample = torch.randn(n, self.z_dim).to(device=device)
        if with_grad:
            samples = self.forward(sample)
        else:
            with torch.no_grad():
                samples = self.forward(sample)
        # ========================
        return samples

    def forward(self, z):
        """
        :param z: A batch of latent space samples of shape (N, latent_dim).
        :return: A batch of generated images of shape (N,C,H,W) which should be
        the shape which the Discriminator accepts.
        """
        # TODO: Implement the Generator forward pass.
        #  Don't forget to make sure the output instances have the same
        #  dynamic range as the original (real) images.
        # ====== YOUR CODE: ======
        z_reshape = z.reshape(z.shape[0], -1, self.featuremap_size, self.featuremap_size)
        x = self.cnn(z_reshape)
        # ========================
        return x


def discriminator_loss_fn(y_data, y_generated, data_label=0, label_noise=0.0):
    """
    Computes the combined loss of the discriminator given real and generated
    data using a binary cross-entropy metric.
    This is the loss used to update the Discriminator parameters.
    :param y_data: Discriminator class-scores of instances of data sampled
    from the dataset, shape (N,).
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :param label_noise: The range of the noise to add. For example, if
    data_label=0 and label_noise=0.2 then the labels of the real data will be
    uniformly sampled from the range [-0.1,+0.1].
    :return: The combined loss of both.
    """
    assert data_label == 1 or data_label == 0
    # TODO:
    #  Implement the discriminator loss. Apply noise to both the real data and the
    #  generated labels.
    #  See pytorch's BCEWithLogitsLoss for a numerically stable implementation.
    # ====== YOUR CODE: ======
    BCEWLoss = torch.nn.BCEWithLogitsLoss(reduction="mean")

    noise = torch.rand_like(y_data) * label_noise - label_noise / 2
    labelDataTargets = torch.ones_like(y_data) * data_label
    loss_data = BCEWLoss(y_data, labelDataTargets + noise)
    noiseGenerated = torch.rand_like(y_generated) * label_noise - label_noise / 2
    labelGeneratedTargets = torch.ones_like(y_generated) * (1 - data_label)
    loss_generated = BCEWLoss(y_generated, (labelGeneratedTargets + noiseGenerated))
    # ========================
    return loss_data + loss_generated


def generator_loss_fn(y_generated, data_label=0):
    """
    Computes the loss of the generator given generated data using a
    binary cross-entropy metric.
    This is the loss used to update the Generator parameters.
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :return: The generator loss.
    """
    assert data_label == 1 or data_label == 0
    # TODO:
    #  Implement the Generator loss.
    #  Think about what you need to compare the input to, in order to
    #  formulate the loss in terms of Binary Cross Entropy.
    # ====== YOUR CODE: ======
    BCEWLoss = torch.nn.BCEWithLogitsLoss(reduction="mean")
    DataLossWeights = torch.ones_like(y_generated) * data_label
    loss = BCEWLoss(y_generated, DataLossWeights)
    # ========================
    return loss


def train_batch(
        dsc_model: Discriminator,
        gen_model: Generator,
        dsc_loss_fn: Callable,
        gen_loss_fn: Callable,
        dsc_optimizer: Optimizer,
        gen_optimizer: Optimizer,
        x_data: Tensor,
):
    """
    Trains a GAN for over one batch, updating both the discriminator and
    generator.
    :return: The discriminator and generator losses.
    """

    # TODO: Discriminator update
    #  1. Show the discriminator real and generated data
    #  2. Calculate discriminator loss
    #  3. Update discriminator parameters
    # ====== YOUR CODE: ======
    sample = gen_model.sample(x_data.shape[0], with_grad=False)
    gen_optimizer.zero_grad()
    dsc_optimizer.zero_grad()
    dsc_real_out = dsc_model(x_data)
    dsc_gen_out = dsc_model(sample)
    dsc_loss = dsc_loss_fn(dsc_real_out, dsc_gen_out)
    dsc_loss.backward()
    dsc_optimizer.step()
    # ========================

    # TODO: Generator update
    #  1. Show the discriminator generated data
    #  2. Calculate generator loss
    #  3. Update generator parameters
    # ====== YOUR CODE: ======
    sample = gen_model.sample(x_data.shape[0], with_grad=True)
    dsc_gen_out = dsc_model(sample)
    dsc_optimizer.zero_grad()
    gen_optimizer.zero_grad()
    gen_loss = gen_loss_fn(dsc_gen_out)
    gen_loss.backward()
    gen_optimizer.step()
    # ========================

    return dsc_loss.item(), gen_loss.item()


def save_checkpoint(gen_model, dsc_losses, gen_losses, checkpoint_file):
    """
    Saves a checkpoint of the generator, if necessary.
    :param gen_model: The Generator model to save.
    :param dsc_losses: Avg. discriminator loss per epoch.
    :param gen_losses: Avg. generator loss per epoch.
    :param checkpoint_file: Path without extension to save generator to.
    """

    saved = False
    checkpoint_file = f"{checkpoint_file}.pt"

    # TODO:
    #  Save a checkpoint of the generator model. You can use torch.save().
    #  You should decide what logic to use for deciding when to save.
    #  If you save, set saved to True.
    # ====== YOUR CODE: ======
    if len(gen_losses) % 4 == 0:
        torch.save(gen_model, checkpoint_file)
        saved = True
    # ========================

    return saved
