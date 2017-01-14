from vgg19 import VGG19
import theano.tensor as T
import lasagne
import theano
import numpy as np
from lasagne.utils import floatX
import loss
import optimize
import time
import tiling


def get_layer_activations(layers, image):
    input_im_theano = T.tensor4()
    outputs = lasagne.layers.get_output(layers.values(), input_im_theano)
    return {k: theano.shared(output.eval({input_im_theano: image}))
            for k, output in zip(layers.keys(), outputs)}


def get_layer_activations_for_image(layer_names, image):
    # image.shape is bc01 (b, c, h, w)
    net = VGG19(image.shape[2], image.shape[3])
    net.load_params('vgg19_normalized.pkl')
    layers = net.get_layers(layer_names)
    return get_layer_activations(layers, image)


def generate_stylized_image(content, style, content_weight, style_weight, style_layer_weight_factor, total_variation_weight, optimizer_iterations,
                            optimizer_checkpoint, init_random, init_random_content, init_image, print_loss):
    stylize = Stylize(content, True, style, content_weight, style_weight, style_layer_weight_factor, total_variation_weight, print_loss, print_progress=True)
    xs, loss_val = stylize.generate(content, generate_initial_generated_image(content, init_random, init_random_content, init_image), optimizer_iterations, optimizer_checkpoint)
    return xs


def generate_stylized_image_tiled(content, style, content_weight, style_weight, style_layer_weight_factor, total_variation_weight, optimizer_iterations,
                                  optimizer_checkpoint, init_random, init_random_content, init_image, tile_size, print_loss):
    tile_shape = (1, 3, tile_size, tile_size)

    init_generated_image = generate_initial_generated_image(content, init_random, init_random_content, init_image)

    content_idxs = tiling.make_img_idx_from_canvas(content.shape, tile_shape)
    # print content_idxs
    result_xs = []

    content_canvas = tiling.pad_image_for_tiling(content, tile_shape)
    generated_canvas = tiling.pad_image_for_tiling(init_generated_image, tile_shape)
    inner_iters = 30 # fewer than this, and the results get pretty bad
    stylize = Stylize(generated_canvas[:, :, :tile_shape[2], :tile_shape[3]], False, style, content_weight, style_weight, style_layer_weight_factor, total_variation_weight, print_loss, print_progress=False)
    feather_filter = np.pad(np.ones((2, 2)), tile_size // 2 - 1, mode='linear_ramp', end_values=0.)
    overall_start_time = time.time()
    start_time = overall_start_time

    outer_iters = optimizer_iterations // inner_iters
    num_tiles = sum(1 for _ in tiling.make_tile_indexes_from_canvas(generated_canvas.shape, tile_shape))
    for it in range(outer_iters):
        next_generated_canvas = np.zeros_like(generated_canvas)
        for j, (img, ix) in enumerate(zip(tiling.make_tiles_from_canvas(generated_canvas, tile_shape), tiling.make_tile_indexes_from_canvas(generated_canvas.shape, tile_shape))):
            xs, loss_val = stylize.generate(content_canvas[ix], img, inner_iters, inner_iters)
            next_generated_canvas[ix] += xs[-1] * feather_filter
            print('iteration %d' % (it * num_tiles + j + 1,))
            print('Current loss value: %f' % (loss_val,))
            end_time = time.time()
            print('Iteration %d completed in %fs' % (it * num_tiles + j + 1, end_time - start_time))
            start_time = end_time

        result_xs.append(next_generated_canvas[content_idxs])
        generated_canvas = next_generated_canvas

    overall_end_time = time.time()
    print('Optimization completed in %fs' % (overall_end_time - overall_start_time,))

    return result_xs


def generate_initial_generated_image(content, init_random, init_random_content, init_image):
    # Initialize with the content image
    if init_image is not None:
        return init_image.copy()
    elif init_random:
        return np.random.uniform(-128., 128., content.shape)
    elif init_random_content:
        return content + np.random.normal(loc=0.0, scale=np.std(content) * 0.1, size=content.shape)
    else:
        return content.copy()


class Stylize:
    def __init__(self, content, content_is_const, style, content_weight, style_weight, style_layer_weight_factor, total_variation_weight, print_loss, print_progress):
        self.content_is_const = content_is_const
        self.style = style
        self.net = VGG19(content.shape[2], content.shape[3])
        self.net.load_params('vgg19_normalized.pkl')
    
        style_layer_names = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
        style_layers = self.net.get_layers(style_layer_names)
        content_layer_names = ['conv4_2']
        content_layers = self.net.get_layers(content_layer_names)
        all_layers = content_layers.copy()
        all_layers.update(style_layers)
    
        # expression for layer activations for content
        if self.content_is_const:
            self.content = content
        else:
            self.content = theano.shared(content)
        content_features = self.get_layer_activations_for_image(content_layer_names, content)
        # Pre-compute layer activations for style
        style_features = self.get_layer_activations_for_image(style_layer_names, self.style)
    
        # Get expressions for layer activations for generated image
        self.generated_image = theano.shared(floatX(np.zeros(shape=(1, 3, content.shape[2], content.shape[3]))))
    
        generated_features = {k: v for k, v in zip(all_layers.keys(), lasagne.layers.get_output(all_layers.values(), self.generated_image))}
    
        total_loss = loss.total_loss(content_features, style_features, generated_features, self.generated_image, content_layer_names,
                                     style_layer_names, content_weight, style_weight, style_layer_weight_factor, total_variation_weight, print_loss)
    
        grad = T.grad(total_loss, self.generated_image)
    
        # Theano function to evaluate loss and gradient
        self.f_outputs = theano.function([], [total_loss, grad])
        self.print_progress = print_progress
        
    def get_layer_activations_for_image(self, layer_names, image):
        # image.shape is bc01 (b, c, h, w)
        layers = self.net.get_layers(layer_names)
        return get_layer_activations(layers, image)
    
    def generate(self, content, init_image, optimizer_iterations, optimizer_checkpoint):
        if not self.content_is_const:
            self.content.set_value(content)
        self.generated_image.set_value(init_image)
    
        # scipy optimize requires that the parameters are of type float64
        x0 = self.generated_image.get_value().astype('float64')
    
        # our record. Start with the style and the content.
        style2 = self.style[:, :, :x0.shape[2], :x0.shape[3]]
        if style2.shape[2] < x0.shape[2]:
            style2 = np.concatenate((style2, np.zeros((x0.shape[0], x0.shape[1], x0.shape[2] - self.style.shape[2], style2.shape[3]))), axis=2)
        if style2.shape[3] < x0.shape[3]:
            style2 = np.concatenate((style2, np.zeros((x0.shape[0], x0.shape[1], style2.shape[2], x0.shape[3] - self.style.shape[3]))), axis=3)
    
        xs = [content, style2, x0]
    
        overall_start_time = time.time()
        last_loss_val = 0.
        for x, i, loss_val, iter_duration in optimize.optimize(x0, self.generated_image, self.f_outputs, num_iterations=optimizer_iterations, checkpoint_iterations=optimizer_checkpoint):
            if self.print_progress:
                print('iteration %d' % (i,))
                print('Current loss value: %f' % (loss_val,))
                print('Iteration %d completed in %fs' % (i, iter_duration))
            xs.append(x)
            last_loss_val = loss_val

        if self.print_progress:
            overall_end_time = time.time()
            print('Optimization completed in %fs' % (overall_end_time - overall_start_time,))
    
        return xs, last_loss_val
