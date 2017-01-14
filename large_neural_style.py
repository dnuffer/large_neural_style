from __future__ import print_function
import skimage.transform
from argparse import ArgumentParser

import img
import stylize
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import matplotlib.pyplot as plt
plt.interactive(False)

from img import resize_image_to_vgg_input, save_vgg_to_image_file, image_to_shape, show_vgg_image, compute_shape, \
    image_to_vgg_input, vgg_input_to_image
import numpy as np


def main():
    argParser = ArgumentParser(description='Large Neural Style')
    argParser.add_argument('--content', required=True, help='Content image path')
    argParser.add_argument('--style', required=True, help='Style image path')
    argParser.add_argument('--output', required=True, help='Generated image path')
    argParser.add_argument('--content-weight', type=float, default=1e2, help='Weight of content penalty (100.0 default)')
    argParser.add_argument('--style-weight', type=float, default=1e0, help='Weight of style penalty (1.0 default)')
    argParser.add_argument('--style-layer-weight-factor', type=float, default=3.0,
                           help='Multiplicative factor for the style-layers. Applied in layer order (3.0 default)')
    argParser.add_argument('--total-variation-weight', type=float, default=1e-1,
                           help='Weight of total variation penalty (0.1 default)')
    argParser.add_argument('--optimizer-iterations', type=int, default=100,
                           help='The number of iterations to run the optimizer per phase. When pyramid is enabled, 5x '
                                'as many is done for the smallest phase.')
    argParser.add_argument('--optimizer-checkpoint', type=int, default=50,
                           help='Print out progress this many iterations. Do not set lower than 30, as that will '
                                'degrade the quality')
    argParser.add_argument('--max-side', type=int, default=1920,
                           help='Specify the dimension of the largest size of the generated image. The aspect ratio of '
                                'the content image is preserved, which is how the size of the smaller dimension is '
                                'calculated (default 1920)')
    argParser.add_argument('--style-scale', type=float, default=1.0,
                           help='The style image is resized to <style-scale> * <output resolution> (1.0 default)')
    argParser.add_argument('--display-results', dest='display_results', action='store_true',
                           help='Display the results (off by default)')
    argParser.add_argument('--no-display-results', dest='display_results', action='store_false',
                           help='Do not display the results (default)')
    argParser.set_defaults(display_results=False)
    argParser.add_argument('--init-content', dest='init_content', action='store_true',
                           help='Start optimizing with the content image. This typically generates an image close to '
                                'the content image. (on by default)')
    argParser.set_defaults(init_content=True)
    argParser.add_argument('--init-random', dest='init_random', action='store_true',
                           help='Start optimizing from randomly generated pixels. This typically generates an image '
                                'with stronger style (off by default)')
    argParser.set_defaults(init_random=False)
    argParser.add_argument('--init-random-content', dest='init_random_content', action='store_true',
                           help='Start optimizing from the content with added random gaussian noise '
                                '(sigma=0.1 * stddev(image)) (off by default)')
    argParser.set_defaults(init_random_content=False)
    argParser.add_argument('--tile', dest='tile', action='store_true',
                           help='Use tiling algorithm to generate high resolution images (on by default)')
    argParser.add_argument('--no-tile', dest='tile', action='store_false', help='Do not use tiling algorithm')
    argParser.set_defaults(tile=True)
    argParser.add_argument('--tile-size', type=int, default=224 * 4,
                           help='The size of the tile to use. This affects how much memory is needed. 896 (224*4) is '
                                'safe to use with 12GB. With 4GB, 448 is good. Larger tile size generates better '
                                'results')  # VGG is trained on 224x224 images, go for a multiple of that. The larger
    # this is, the more syle can be captured. * 5 takes about 10 GB
    argParser.add_argument('--pyramid', dest='pyramid', action='store_true',
                           help='Use the pyramid algorithm (on by default)')
    argParser.add_argument('--no-pyramid', dest='pyramid', action='store_false',
                           help='Do not use the pyramid algorithm')
    argParser.set_defaults(pyramid=True)
    argParser.add_argument('--print-loss', action='store_true',
                           help='Print intermediate loss values during execution. Useful for debugging and '
                                'understanding how they affect the algorithm (off by default)')
    argParser.set_defaults(print_loss=False)
    args = argParser.parse_args()

    if args.init_content:
        args.init_random = False
        args.init_random_content = False

    xs = []
    content_image = plt.imread(args.content)
    print('Content image shape: ', content_image.shape)

    if args.pyramid:
        shapes = []
        cur_shape = compute_shape(content_image.shape, args.max_side)
        while max(cur_shape[0], cur_shape[1]) > 224:
            shapes = [cur_shape] + shapes
            cur_shape = (cur_shape[0] // 2, cur_shape[1] // 2, cur_shape[2])

        print('Pyramid sizes:', shapes)
        last_generated_image = None
        for i, shape in enumerate(shapes):
            print('shape:', shape)
            resized_content_image = img.add_reflect_padding(skimage.transform.resize(content_image, shape, preserve_range=True))
            print('resized_content_image.shape: ', resized_content_image.shape)
            if i > 0:
                last_image = img.remove_reflect_padding(vgg_input_to_image(last_generated_image))
                print('last_image.shape:', last_image.shape)
                last_image_resized = skimage.transform.resize(last_image, shape, preserve_range=True)
                print('last_image_resize.shape:', last_image_resized.shape)
                last_generated_image = image_to_vgg_input(img.add_reflect_padding(last_image_resized))
                print('last_generated_image.shape:', last_generated_image.shape)
            content = image_to_vgg_input(resized_content_image)
            print('content.shape:', content.shape)
            style = resize_image_to_vgg_input(plt.imread(args.style), int(max(shape) * args.style_scale))
            print('style.shape:', style.shape)
            # make the style dimensions match the content by cropping and padding w/reflection
            style = image_to_shape(style, (content.shape[0], content.shape[1], content.shape[2] - 64, content.shape[3] - 64))
            print('after image_to_shape(), style.shape:', style.shape)

            # TODO: if the shape is too big (catch MemoryError?) use tiling

            # replace init_random after the first iteration so that subsequent iterations use the resized previous iteration's result
            init_random = args.init_random if i == 0 else False
            init_random_content = args.init_random_content if i == 0 else False
            optimizer_iterations = args.optimizer_iterations * 5 if i == 0 else args.optimizer_iterations
            xs = stylize.generate_stylized_image(content, style, args.content_weight, args.style_weight,
                                                 args.style_layer_weight_factor, # / float(i + 1), # * float(i + 1),
                                                 args.total_variation_weight,
                                                 optimizer_iterations,
                                                 args.optimizer_checkpoint,
                                                 init_random,
                                                 init_random_content,
                                                 last_generated_image,
                                                 args.print_loss)
            xs = [img.remove_reflect_padding_vgg(x) for x in xs]
            print('after remove reflect padding, xs[-1].shape:', xs[-1].shape)
            combined_image = np.concatenate(xs[::-1], axis=3)
            print('combined_image.shape:', combined_image.shape)
            show_vgg_image(combined_image)
            last_generated_image = xs[-1]
            print('last_generated_image.shape: ', last_generated_image.shape)
            print('end of loop')


    else:
        content = resize_image_to_vgg_input(content_image, args.max_side)
        if args.tile:
            if args.tile_size % 2 != 0:
                raise Exception("tile_size is not a multiple of 2")

            style_image = plt.imread(args.style)
            content_style_size_ratio = (content_image.shape[0] + content.shape[1]) / (style_image.shape[0] + style_image.shape[1])
            style = resize_image_to_vgg_input(style_image, int(args.max_side * args.style_scale))
            # make the style dimensions match the content by cropping and padding w/reflection
            style = image_to_shape(style, (1, 3, args.tile_size, args.tile_size))
            print(style.shape)

            xs = stylize.generate_stylized_image_tiled(content, style, args.content_weight, args.style_weight,
                                                       args.style_layer_weight_factor,
                                                       args.total_variation_weight,
                                                       args.optimizer_iterations,
                                                       args.optimizer_checkpoint,
                                                       args.init_random,
                                                       args.init_random_content,
                                                       None,
                                                       args.tile_size,
                                                       args.print_loss)
        else:
            style = resize_image_to_vgg_input(plt.imread(args.style), int(args.max_side * args.style_scale))
            # make the style dimensions match the content by cropping and padding w/reflection
            style = image_to_shape(style, content.shape)

            xs = stylize.generate_stylized_image(content, style, args.content_weight, args.style_weight,
                                                 args.style_layer_weight_factor,
                                                 args.total_variation_weight,
                                                 args.optimizer_iterations,
                                                 args.optimizer_checkpoint,
                                                 args.init_random,
                                                 args.init_random_content,
                                                 None,
                                                 args.print_loss)

    # save the result
    if args.output is not None:
        save_vgg_to_image_file(args.output, xs[-1])

    # display the result
    if args.display_results:
        combined_image = np.concatenate(xs[::-1], axis=3)
        show_vgg_image(combined_image)

    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main())
