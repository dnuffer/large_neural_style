from lasagne.layers import InputLayer, Pool2DLayer, set_all_param_values
try:
    from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
except:
    from lasagne.layers import Conv2DLayer as ConvLayer
import pickle

# uses average pooling instead of maxpooling

class VGG19:
    def __init__(self, height, width):
        self.net = {}
        self.net['input'] = InputLayer((1, 3, height, width))
        self.net['conv1_1'] = ConvLayer(self.net['input'], 64, 3, pad=1, flip_filters=False)
        self.net['conv1_2'] = ConvLayer(self.net['conv1_1'], 64, 3, pad=1, flip_filters=False)
        self.net['pool1'] = Pool2DLayer(self.net['conv1_2'], 2, mode='average_exc_pad')
        self.net['conv2_1'] = ConvLayer(self.net['pool1'], 128, 3, pad=1, flip_filters=False)
        self.net['conv2_2'] = ConvLayer(self.net['conv2_1'], 128, 3, pad=1, flip_filters=False)
        self.net['pool2'] = Pool2DLayer(self.net['conv2_2'], 2, mode='average_exc_pad')
        self.net['conv3_1'] = ConvLayer(self.net['pool2'], 256, 3, pad=1, flip_filters=False)
        self.net['conv3_2'] = ConvLayer(self.net['conv3_1'], 256, 3, pad=1, flip_filters=False)
        self.net['conv3_3'] = ConvLayer(self.net['conv3_2'], 256, 3, pad=1, flip_filters=False)
        self.net['conv3_4'] = ConvLayer(self.net['conv3_3'], 256, 3, pad=1, flip_filters=False)
        self.net['pool3'] = Pool2DLayer(self.net['conv3_4'], 2, mode='average_exc_pad')
        self.net['conv4_1'] = ConvLayer(self.net['pool3'], 512, 3, pad=1, flip_filters=False)
        self.net['conv4_2'] = ConvLayer(self.net['conv4_1'], 512, 3, pad=1, flip_filters=False)
        self.net['conv4_3'] = ConvLayer(self.net['conv4_2'], 512, 3, pad=1, flip_filters=False)
        self.net['conv4_4'] = ConvLayer(self.net['conv4_3'], 512, 3, pad=1, flip_filters=False)
        self.net['pool4'] = Pool2DLayer(self.net['conv4_4'], 2, mode='average_exc_pad')
        self.net['conv5_1'] = ConvLayer(self.net['pool4'], 512, 3, pad=1, flip_filters=False)
        self.net['conv5_2'] = ConvLayer(self.net['conv5_1'], 512, 3, pad=1, flip_filters=False)
        self.net['conv5_3'] = ConvLayer(self.net['conv5_2'], 512, 3, pad=1, flip_filters=False)
        self.net['conv5_4'] = ConvLayer(self.net['conv5_3'], 512, 3, pad=1, flip_filters=False)
        self.net['pool5'] = Pool2DLayer(self.net['conv5_4'], 2, mode='average_exc_pad')

    def load_params(self, weights_filename):
        values = pickle.load(open(weights_filename))['param values']
        set_all_param_values(self.net['pool5'], values)

    def get_layers(self, layer_names):
        return {k: self.net[k] for k in layer_names}