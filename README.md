# SSL-2023
Small group project in Self supervised learning

## Goal

The goal for this project is to build a simple Denoising Diffusion Probabilistic Model (DDPM) to generate images based on the MNIST dataset. 
We will originially only train a self-supervised model, i.e. ignoring the labels. We might later condition the model on the labels in order to 
generate a hand-drawn version of a given digit. This would be a semi-supervised approach. 

## DDPMs
Denoising Diffusion Probabilistic Models generate new samples that fit in a distribution by following a diffusion process. This is a markov chain where each step removes some noise from the last. The prediction markov chain is typically referred to as the reverse process. To generate sample a forward process takes a sample from the original distribution (for example an image) and gradually adds noise to it. the process is illustrated in the below image. 

![Diffusion process](diffusion.png )
(Image credits: [cvpr2022](https://cvpr2022-tutorial-diffusion-models.github.io/))

## Model

The model we train is a simple U-Net model. It takes as input a noisy image and a timestep along the diffusion process. Based on this it outputs a prediction on which noise has been added to the image. 


## Dataset / Example generation
The dataset we are using for this task is MNSIT handwritten digits. This dataset is labeled, but to keep with the spirit of the SSL-task we just use the images. These images are normalized to [-1, 1].
To create an example for the network to train on we do the following.
We take an unaltered image and noise with the same dimension as the image sampled from a normal distribution and create a interpolation between the two images. A parameter time_step decides how much weight is assigned to the original image and how much is assigned to the noise.
The resulting noisy image and the time_step will be the input to the network and the sampled noise is the target.

## Training

The following hardware was used for training:

- CPU: AMD Ryzen 7 3700X
- GPU: Nvidia RTX 2080ti, 11GB VRAM
- RAM: 32GB

The model was trained for 100 epochs with a batch size of 512, an Adam optimizer with learning rate of `0.001`.

Loss plateaued as soon as around 20 epochs.

The total training took around 5.5 minutes for 20 epochs and around 30 minutes for 100 epochs.


## Sampling
To sample an image we start from normal distribution noise with the same dimentions as our training data.
We then remove noise in each iteration until we end up with what hopefully should look like an image belonging to our dataset.

In more detail:
In each iteration the network predicts the noise in the image and a fraction of this predicted noise is removed.
Then a small amount of noise from a new normal distribution is added back to the image.
In the last iteration no noise is added to the image.
Both the noise added to the image and how much noise is removed is decided by hyperparameters called the noise schedule.



## Results

Example result from the model, also showing the steps going from random noise to the sample: 

![Example result](fig_readme.png)

## How to run 

First, dependencies must be installed: 
```bash
pip install -r requirements.txt
```

To train a model run 
```bash 
python model.py train
```

Then, to generate examples use

```bash
python model.py predict -l model
```
