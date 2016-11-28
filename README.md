## Learning to Generate Samples from Noise Through Infusion Training
Reproducing an experiment from the ICLR 2017 submission by Bordes et al. from MILA.

### How to run
Currently, only the MNIST experiments are available.

```[bash]
> python infusion.py
```

would give you the following generation after 30 epochs of training.

It is nice to see that all the numbers are generated, but it is not known whether it has the same "mode avoiding" problem as GAN.

![](generation.png)

### Observations
The training is robust (in that not much tuning is required) but not very stable.
Occasionally the loss will rise suddenly, and the model will generate artifacts.
The cause of the problem is yet unknown.

Also, the generator tends to generate '8' that look like '1's.
