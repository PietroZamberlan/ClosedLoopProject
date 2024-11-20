# Document for tracking GP code progres and missing features

For now the GP works almost without ever encountering numerical problems, the utility estimation though is still too slow.

- [ ] Problems emerge when no E step is done and only hyperparameters are updated.

- [ ] The KL divergence presents some spikes at certain moments of the training that are not there when only optimising the variational parameters. 
        There is a problem with the hyperparameters optimization. 

For a detailed list of tasks and progress, see the [TODO.md](../TODO.md) file.