# Generative-Modelling

Generative Modelling project

## Introduction

This repository contains code for a generative diffusion model. This model is composed of a forward process (diffusion process) and a reverse process. 
The forward process gradually transforms inputs from their natural population distribution to a Gaussian distribution. 
The reverse process is designed to train a model to slowly convert a sample from a Gaussian distribution back to the input distribution.

## Key Concepts

- **Forward Process (Diffusion Process)**: This process takes the natural population distribution and gradually transforms it into a Gaussian distribution. It does so through a series of steps, and at each step, noise is added to the data. Over time, the data approaches a Gaussian distribution.

- **Reverse Process**: The reverse process is trained to perform the reverse transformation. It takes a sample from a Gaussian distribution and slowly converts it back to the original input distribution.



