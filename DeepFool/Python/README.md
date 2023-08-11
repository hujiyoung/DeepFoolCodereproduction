# DeepFool
DeepFool is a simple algorithm to find the minimum adversarial perturbations in deep networks. And now we test is on the model ResNet18 and on the CIFAR-10 dataset.

### deepfool.py

This function implements the algorithm proposed in [[1]](http://arxiv.org/pdf/1511.04599) using PyTorch to find adversarial perturbations.

__Note__: The final softmax (loss) layer should be removed in order to prevent numerical instabilities. Because the `zero_gradients` is too old, we discard it and instead use a simple `x.grad.zero_()`.

The parameters of the function are:

- `image`: Image of size `HxWx3d`
- `net`: neural network (input: images, output: values of activation **BEFORE** softmax).
- `num_classes`: limits the number of classes to test against, by default = 10.
- `max_iter`: max number of iterations, by default = 50.

### test_deepfool.py

We want to test DeepFool our model and dataset, we first call function `deepfool` in the deepfool.py to create adversarial example, then we calculate the accuracy of our model on the adversarial example. Finally, we display the adversarial example.

## Reference
[1] S. Moosavi-Dezfooli, A. Fawzi, P. Frossard:
*DeepFool: a simple and accurate method to fool deep neural networks*.  In Computer Vision and Pattern Recognition (CVPR ’16), IEEE, 2016.
