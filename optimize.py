from lasagne.utils import floatX
import time
import scipy
import numpy as np

# this LossAndGradSimultaneousEvaluator class makes it possible
# to compute loss and gradients in one pass
# while retrieving them via two separate functions,
# "loss" and "grads". This is done because scipy.optimize
# requires separate functions for loss and gradients,
# but computing them separately would be inefficient.
class LossAndGradSimultaneousEvaluator(object):
    def __init__(self, height, width, generated_image, f_outputs):
        self.f_outputs = f_outputs
        self.generated_image = generated_image
        self.loss_value = None
        self.grad_values = None
        self.height = height
        self.width = width

    def eval_loss_and_grads(self, x):
        x = x.reshape((1, 3, self.height, self.width))
        self.generated_image.set_value(floatX(x))
        outs = self.f_outputs()
        loss_value = outs[0]
        grad_values = outs[1].flatten().astype('float64')
        return loss_value, grad_values

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = self.eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values


def optimize(x0, generated_image, f_outputs, num_iterations, checkpoint_iterations):
    evaluator = LossAndGradSimultaneousEvaluator(x0.shape[2], x0.shape[3], generated_image, f_outputs)
    for i in range(num_iterations // checkpoint_iterations + 1):
        start_time = time.time()
        remaining_iterations = num_iterations - (checkpoint_iterations * i)
        if remaining_iterations > 0:
            x, min_val, info = scipy.optimize.fmin_l_bfgs_b(evaluator.loss, x0.flatten(), fprime=evaluator.grads, maxfun=min(remaining_iterations, checkpoint_iterations))
            x0 = generated_image.get_value().astype('float64')
            end_time = time.time()
            yield (x0, (i + 1) * checkpoint_iterations, min_val, end_time - start_time)

