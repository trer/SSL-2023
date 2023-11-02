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


## Training


## Sampling


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
