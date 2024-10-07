# Closed Loop Project

## Overview
This project allows running a closed loop EMA256 experiment with images a stimulation, using a Gaussian process algorithm to select the following stimulation to be presented. 

In this first version GPs are trained to predict the responses of a sngle unit of the MEA, not necessarely a cell.

The pipeline works as follows:

main.py is the driver script that executes all of the necessary actions in order, these actions, are, in short:

- Selecting the choosen channel and initial hyperparameters as given by a first analysis of the retina on the Windows PC,
- Generating a VEC file for showing 50 random images using the DMD,
- Collecting the responses to these 50 images and using them to training a GP
- Evaluate utility for each of the remaining images, chose the most useful one
- Generate VEC file for that image
- Starting a listening client with TCP/listener_linux for receiving packets

TCP/listener_linux.py is executed, having the linux machine wait for packets

## Components
- [Gaussian process code](docs/GP.md)
- [TCP protocol code](docs/TCP.md)
- [DMD code](docs/DMD.md)