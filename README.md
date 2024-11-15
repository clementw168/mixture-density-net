# A reimplementation of Mixture Density Networks

This repository reimplements the mixture density network (MDN) model proposed by Bishop (1994) in PyTorch. The MDN model is a neural network that can predict multiple possible outputs for a given input. Contrary to traditional neural networks for regression trained with mean squared error, the MDN model is trained with maximum likelihood estimation with a Gaussian mixture distribution.

The project evaluates the capability of the MDN model to predict multiple possible outputs for a given input. Three datasets were considered :
- a toy ill-posed sinusoidal inverse problem,
- robot kinematics,
- MNIST.

Our report is available [here](assets/report.pdf).

You can also checkout our [poster](assets/poster.pdf).

![poster](assets/poster.jpg)
