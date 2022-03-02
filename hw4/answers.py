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
        batch_size=16,
        z_dim=32,
        data_label=1,
        label_noise=0.05,
        discriminator_optimizer=dict(
            type="Adam",  # Any name in nn.optim like SGD, Adam
            lr=0.0001,
            # You an add extra args for the optimizer here
            betas=(0.5, 0.9),
        ),
        generator_optimizer=dict(
            type="Adam",  # Any name in nn.optim like SGD, Adam
            lr=0.0001,
            # You an add extra args for the optimizer here
            betas=(0.5, 0.9),
        ),
    )
    # TODO: Tweak the hyperparameters to train your GAN.
    # ====== YOUR CODE: ======
    #raise NotImplementedError()
    # ========================
    return hypers


part3_q1 = r"""
**Your answer:**

we need to do so when we are training the discriminator. when we do so, we wouldn't want that the calculations of the gradients
would be done on the generator - if it would happen, this will change the parameters of the generator for the worse - 
this would happen du to the fact that in order to decrese the loss of the discriminator we need worsen the generator.
and when training the generator we once again need thous gradients - in order to train the generator.
"""

part3_q2 = r"""
**Your answer:**
1. During training, both the generator and the discriminator are being trained together.
This entails that the discriminator is also being trained and improved "as we go" which means that the loss
of the generator at a certain point has much less meaning by itself.
2. the meaning of this is that the generator has learned to create example that causes the discrimnator to always 
provide a false prediction. and becuse of this, there is no gradients that allow the discriminator to learn, while the generator
is still improving and getting better.
 
"""

part3_q3 = r"""
**Your answer:**
we can see that in vae the result looks more like the samples that were given to the model. when in the gan model the 
results were more abstract then accurate duplicates of the original data set. this can be explained by the way the gan model is designed -
when the generator learns to overcome the discriminator its probebly learns more general features of the shape than the
 exact replica of the original dataset - what generates a less similar result of the image. another key point is that the
 time that takes for the gan model to train is longer. 
"""

# ==============


# ==============
# Part 4 answers
# ==============


def part4_affine_backward(ctx, grad_output):
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
