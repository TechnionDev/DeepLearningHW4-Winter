r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""
import torch

# ==============
# Part 1 answers



def part1_pg_hyperparams():
    hp = dict(
        batch_size=32, gamma=0.99, beta=0.5, learn_rate=1e-3, eps=1e-8, num_workers=2,
    )
    # TODO: Tweak the hyperparameters if needed.
    #  You can also add new ones if you need them for your model's __init__.
    # ====== YOUR CODE: ======
    hp['batch_size'] = 16
    #hp['gamma'] = 0.99
    #hp['beta'] = 0.5
    #hp['learn_rate'] = 1e-3
    #hp['eps'] = 1e-8
    hp['num_workers'] = 0
    # ========================
    return hp


def part1_aac_hyperparams():
    hp = dict(
        batch_size=32,
        gamma=0.99,
        beta=1.0,
        delta=1.0,
        learn_rate=1e-3,
        eps=1e-8,
        num_workers=2,
    )
    # TODO: Tweak the hyperparameters. You can also add new ones if you need
    #   them for your model implementation.
    # ====== YOUR CODE: ======
    hp['batch_size'] = 8
    #hp['gamma'] = 0.99
    #hp['beta'] = 1.0
    #hp['delta'] = 1.0
    #hp['learn_rate'] = 1e-3
    #hp['eps'] = 1e-8
    hp['num_workers'] = 0
    # ========================
    return hp


part1_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part1_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part1_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0, h_dim=0, z_dim=0, x_sigma2=0, learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    hypers['batch_size'] = 64
    hypers['h_dim'] = 256
    hypers['z_dim'] = 16
    hypers['learn_rate'] = 0.0001
    hypers['betas'] = 0.9, 0.9
    hypers['x_sigma2'] = 0.001
    # ========================
    return hypers


part2_q1 = r"""
**Your answer:**
The sigma^2 is used for regularization purposes - std equal to one. the purpose of this regularization is
to be applied on the data term vs. the Kullback-Liebler divergence loss. by adding this regularization we 
encouraging or discouraging the loss in order to minimize the distance between the original image and its enc-dec.
"""

part2_q2 = r"""
**Your answer:**
1. the first term (the data term) represents the distance  between an original encoding of a specific image, to 
 the decoding-encoding output of that image.it would be minimized when the image corresponds directly
to the decoding of its encoding.

the second part correlates to the distance of the probability distribution of the encoder vs, the normal gaussian distribution
which we assume is the latent space z. 
this part of the equations will help us sample from the latent space Z and generate fake images.

2. the latent space dist would be trained to be closer to the normal gaussian distribution, therefore, creating a more 
"meaningful" encoded results such that the decoder would get a more meaningful data space it can train on.
3. the benefits are that our latent space will generate a more meaningful  space of distribution for our decoder model.
"""

part2_q3 = r"""
**Your answer:**
modeling the similarity of unknown results to the spesipic problem domain the hard, and there is no easy way to do so.
instead our goal should be to model the probability of a result to be as real as possible - 
and by doing that, this would maximise the evidence distribution.
"""

part2_q4 = r"""
**Your answer:**

as we know, the value of sigma2 should be between 0 and 1.
but due to vanishing gradients we would encounter with numerical problems - so in order to ensure that the 
value of sigma2 would be in the correct range - we are using log function; that gives us values of minus inf to 0.

"""

# ==============

# ==============
# Part 3 answers

PART3_CUSTOM_DATA_URL = None


def part3_gan_hyperparams():
    hypers = dict(
        batch_size=0,
        z_dim=0,
        data_label=0,
        label_noise=0.0,
        discriminator_optimizer=dict(
            type="",  # Any name in nn.optim like SGD, Adam
            lr=0.0,
            # You an add extra args for the optimizer here
        ),
        generator_optimizer=dict(
            type="",  # Any name in nn.optim like SGD, Adam
            lr=0.0,
            # You an add extra args for the optimizer here
        ),
    )
    # TODO: Tweak the hyperparameters to train your GAN.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return hypers


part3_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============


# ==============
# Part 4 answers
# ==============


def part4_affine_backward(ctx, grad_output):
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
