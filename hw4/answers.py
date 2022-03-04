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
    hp['beta'] = 0.5
    hp['num_workers'] = 0
    # ========================
    return hp


part1_q1 = r"""
**Your answer:**

By removing dependencies and reducing the variance of the policy gradient, we sub the average expected reward from the current state.
When we subtract the baseline ,the remain is the advantage, we change the rewards to a relative value, measuring how successful a current action is, but not adding importance to the absolute value.
With baseline value, variance is much lower than without. With baseline value we measure the amount of how much an action is better by comparing it to its opposite actions. In that way, 
we increase the different we have between bad and good actions. (that gives lower variance).
An example where it helps us is example that the scores will be positive. 
Increasing the scores of each action is the only way to change the probabilities of the results.
With baseline we will get a distribution where, without a baseline, we will get a distribution where entropy is very low, 
while with a baseline, the distribution will be nearly equal. However, in that way the better action will be rewarded.

"""


part1_q2 = r"""

$v_\pi(s) = q_\pi(s,a) - advantage$, when advantage says the amount of how much an action is better by comparing it to its opposite actions.
q is the value of the state when we choose a specific action, and v is the expectation value of the state when we consider all of the possible action.
The critic produces a score to see which action he decides is most advantageous, and forces the player to go that route.
q-values let us see which action leads to better actions afterwards, by using the history of those q values. 
(q-values approximate the environment that the critic uses to see what actions are better)
We want that action that is better will get a bigger weight,differences between q and v depend on how good the action is 
(better action means a bigger difference between q and v; vice versa). 





"""


part1_q3 = r"""
**Your answer:**

1.
Loss_p:
By using baseline loss, the policy variance is very little because the graph is almost around zero. in that way it makes the loss of th function to have just a tiny effect.

Baseline:
As can be seen, entropy loss does not affect the baseline loss graph, and we can see that both of them are performing almost identically.
We can see that at some point the spg falls (maybe the entropy damages the baseline and therefore damaging performance)

loss_e: 
we get an entropy loss that is lower by combining baseline and entropy.

Mean reward:
we can see that the entropy and non entropy results are almost identical, some std going both ways, and the baseline graphs are becoming better.


2.The actor critic graph trains very quickly to be at a very high performance point,
 but has some variance and noise. We can see in the Loss_P graph that we reach to a point and stay almost stable (to a positive policy loss). 
 In our opinion, this might suggest that the policy route we took is a good one.
 We get the best entropy loss and get a better mean rewards with good results.
 We believe that the more periods the better net we get.


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
    x, w, b = ctx.saved_tensors

    dx = 0.5 * torch.matmul(grad_output, w)
    dw = 0.5 * torch.matmul(grad_output.T, x)
    db = torch.sum(grad_output, axis=0)
    # ========================


    return dx, dw, db
