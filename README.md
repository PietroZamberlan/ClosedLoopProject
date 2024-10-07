# Closed Loop Project

## Overview
This project allows running a closed loop EMA256 experiment with images a stimulation, using a Gaussian process algorithm to select the following stimulation to be presented. 

In this first version GPs are trained to predict the responses of a sngle unit of the MEA, not necessarely a cell.

The pipeline works as follows:

main.py is the driver script that executes all of the necessary actions in order, these actions are, in short:

- Selecting the choosen channel and initial hyperparameters as given by a first analysis of the retina on the Windows PC,
    - Presenting the first 15 minutes of standard stimuli and get the approximate single units STAs. Choose the best one.
- Generating a VEC file for the first 50 random images to show using the DMD
- Initiating zmq TCP connection to the Windows machine, sending the VEC file.
    - Collecting the responses to these 50 images and using them to training a GP with a dedicated TCP - GP script
- Iteratively:
    - Evaluate utility for each of the remaining images, chose the most useful one ( GP utility )
    - Generate VEC file for that image ( DMD utility)
    - Send the created VEC to the DMD and show it
    - Collect response
    - Train GP with the updated training set

TCP/listener_linux.py is executed, having the linux machine wait for packets
## Components
- [Gaussian process code](docs/GP.md)
- [TCP protocol code](docs/TCP.md)
- [DMD code](docs/DMD.md)