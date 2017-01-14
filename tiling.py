"""
>>> image_shape = (1, 3, 3, 4)
>>> tile_shape = (1, 3, 2, 2)
>>> tile_shape_4x4 = (1, 3, 4, 4)
>>> canvas_shape(image_shape, tile_shape)
(1, 3, 5, 6)
>>> canvas_shape(image_shape, (1, 3, 4, 4))
(1, 3, 8, 8)
>>> canvas_shape((1, 3, 3, 5), (1, 3, 4, 4))
(1, 3, 8, 10)
>>> canvas_shape((1, 3, 5, 5), (1, 3, 4, 4))
(1, 3, 10, 10)
>>> a = np.arange(start=1, stop=3*2*3+1).reshape((1, 3, 2, 3))
>>> pad_image(a, ((0, 0), (0, 0), (1,1), (1,1)))
array([[[[ 5,  4,  5,  6,  5],
         [ 2,  1,  2,  3,  2],
         [ 5,  4,  5,  6,  5],
         [ 2,  1,  2,  3,  2]],
<BLANKLINE>
        [[11, 10, 11, 12, 11],
         [ 8,  7,  8,  9,  8],
         [11, 10, 11, 12, 11],
         [ 8,  7,  8,  9,  8]],
<BLANKLINE>
        [[17, 16, 17, 18, 17],
         [14, 13, 14, 15, 14],
         [17, 16, 17, 18, 17],
         [14, 13, 14, 15, 14]]]])
>>> pad_image_for_tiling(a, (1, 3, 2, 2))
array([[[[ 5,  4,  5,  6,  5],
         [ 2,  1,  2,  3,  2],
         [ 5,  4,  5,  6,  5],
         [ 2,  1,  2,  3,  2]],
<BLANKLINE>
        [[11, 10, 11, 12, 11],
         [ 8,  7,  8,  9,  8],
         [11, 10, 11, 12, 11],
         [ 8,  7,  8,  9,  8]],
<BLANKLINE>
        [[17, 16, 17, 18, 17],
         [14, 13, 14, 15, 14],
         [17, 16, 17, 18, 17],
         [14, 13, 14, 15, 14]]]])
>>> a2 = np.arange(start=1, stop=3*3*3+1).reshape((1, 3, 3, 3))
>>> pad_image_for_tiling(a2, (1, 3, 2, 2))
array([[[[ 5,  4,  5,  6,  5],
         [ 2,  1,  2,  3,  2],
         [ 5,  4,  5,  6,  5],
         [ 8,  7,  8,  9,  8],
         [ 5,  4,  5,  6,  5]],
<BLANKLINE>
        [[14, 13, 14, 15, 14],
         [11, 10, 11, 12, 11],
         [14, 13, 14, 15, 14],
         [17, 16, 17, 18, 17],
         [14, 13, 14, 15, 14]],
<BLANKLINE>
        [[23, 22, 23, 24, 23],
         [20, 19, 20, 21, 20],
         [23, 22, 23, 24, 23],
         [26, 25, 26, 27, 26],
         [23, 22, 23, 24, 23]]]])
>>> pad_image_for_tiling(a, (1, 3, 4, 4))
array([[[[ 3,  2,  1,  2,  3,  2,  1,  2],
         [ 6,  5,  4,  5,  6,  5,  4,  5],
         [ 3,  2,  1,  2,  3,  2,  1,  2],
         [ 6,  5,  4,  5,  6,  5,  4,  5],
         [ 3,  2,  1,  2,  3,  2,  1,  2],
         [ 6,  5,  4,  5,  6,  5,  4,  5]],
<BLANKLINE>
        [[ 9,  8,  7,  8,  9,  8,  7,  8],
         [12, 11, 10, 11, 12, 11, 10, 11],
         [ 9,  8,  7,  8,  9,  8,  7,  8],
         [12, 11, 10, 11, 12, 11, 10, 11],
         [ 9,  8,  7,  8,  9,  8,  7,  8],
         [12, 11, 10, 11, 12, 11, 10, 11]],
<BLANKLINE>
        [[15, 14, 13, 14, 15, 14, 13, 14],
         [18, 17, 16, 17, 18, 17, 16, 17],
         [15, 14, 13, 14, 15, 14, 13, 14],
         [18, 17, 16, 17, 18, 17, 16, 17],
         [15, 14, 13, 14, 15, 14, 13, 14],
         [18, 17, 16, 17, 18, 17, 16, 17]]]])
>>> b = np.arange(3*3*4).reshape((1, 3, 3, 4))
>>> pad_image_for_tiling(b, (1, 3, 2, 2))
array([[[[ 5,  4,  5,  6,  7,  6],
         [ 1,  0,  1,  2,  3,  2],
         [ 5,  4,  5,  6,  7,  6],
         [ 9,  8,  9, 10, 11, 10],
         [ 5,  4,  5,  6,  7,  6]],
<BLANKLINE>
        [[17, 16, 17, 18, 19, 18],
         [13, 12, 13, 14, 15, 14],
         [17, 16, 17, 18, 19, 18],
         [21, 20, 21, 22, 23, 22],
         [17, 16, 17, 18, 19, 18]],
<BLANKLINE>
        [[29, 28, 29, 30, 31, 30],
         [25, 24, 25, 26, 27, 26],
         [29, 28, 29, 30, 31, 30],
         [33, 32, 33, 34, 35, 34],
         [29, 28, 29, 30, 31, 30]]]])
>>> pad_image_for_tiling(b, (1, 3, 4, 4))
array([[[[10,  9,  8,  9, 10, 11, 10,  9],
         [ 6,  5,  4,  5,  6,  7,  6,  5],
         [ 2,  1,  0,  1,  2,  3,  2,  1],
         [ 6,  5,  4,  5,  6,  7,  6,  5],
         [10,  9,  8,  9, 10, 11, 10,  9],
         [ 6,  5,  4,  5,  6,  7,  6,  5],
         [ 2,  1,  0,  1,  2,  3,  2,  1],
         [ 6,  5,  4,  5,  6,  7,  6,  5]],
<BLANKLINE>
        [[22, 21, 20, 21, 22, 23, 22, 21],
         [18, 17, 16, 17, 18, 19, 18, 17],
         [14, 13, 12, 13, 14, 15, 14, 13],
         [18, 17, 16, 17, 18, 19, 18, 17],
         [22, 21, 20, 21, 22, 23, 22, 21],
         [18, 17, 16, 17, 18, 19, 18, 17],
         [14, 13, 12, 13, 14, 15, 14, 13],
         [18, 17, 16, 17, 18, 19, 18, 17]],
<BLANKLINE>
        [[34, 33, 32, 33, 34, 35, 34, 33],
         [30, 29, 28, 29, 30, 31, 30, 29],
         [26, 25, 24, 25, 26, 27, 26, 25],
         [30, 29, 28, 29, 30, 31, 30, 29],
         [34, 33, 32, 33, 34, 35, 34, 33],
         [30, 29, 28, 29, 30, 31, 30, 29],
         [26, 25, 24, 25, 26, 27, 26, 25],
         [30, 29, 28, 29, 30, 31, 30, 29]]]])
>>> a5x5 = np.arange(3*5*5).reshape((1, 3, 5, 5))
>>> pad_image_for_tiling(a5x5, (1, 3, 4, 4))
array([[[[12, 11, 10, 11, 12, 13, 14, 13, 12, 11],
         [ 7,  6,  5,  6,  7,  8,  9,  8,  7,  6],
         [ 2,  1,  0,  1,  2,  3,  4,  3,  2,  1],
         [ 7,  6,  5,  6,  7,  8,  9,  8,  7,  6],
         [12, 11, 10, 11, 12, 13, 14, 13, 12, 11],
         [17, 16, 15, 16, 17, 18, 19, 18, 17, 16],
         [22, 21, 20, 21, 22, 23, 24, 23, 22, 21],
         [17, 16, 15, 16, 17, 18, 19, 18, 17, 16],
         [12, 11, 10, 11, 12, 13, 14, 13, 12, 11],
         [ 7,  6,  5,  6,  7,  8,  9,  8,  7,  6]],
<BLANKLINE>
        [[37, 36, 35, 36, 37, 38, 39, 38, 37, 36],
         [32, 31, 30, 31, 32, 33, 34, 33, 32, 31],
         [27, 26, 25, 26, 27, 28, 29, 28, 27, 26],
         [32, 31, 30, 31, 32, 33, 34, 33, 32, 31],
         [37, 36, 35, 36, 37, 38, 39, 38, 37, 36],
         [42, 41, 40, 41, 42, 43, 44, 43, 42, 41],
         [47, 46, 45, 46, 47, 48, 49, 48, 47, 46],
         [42, 41, 40, 41, 42, 43, 44, 43, 42, 41],
         [37, 36, 35, 36, 37, 38, 39, 38, 37, 36],
         [32, 31, 30, 31, 32, 33, 34, 33, 32, 31]],
<BLANKLINE>
        [[62, 61, 60, 61, 62, 63, 64, 63, 62, 61],
         [57, 56, 55, 56, 57, 58, 59, 58, 57, 56],
         [52, 51, 50, 51, 52, 53, 54, 53, 52, 51],
         [57, 56, 55, 56, 57, 58, 59, 58, 57, 56],
         [62, 61, 60, 61, 62, 63, 64, 63, 62, 61],
         [67, 66, 65, 66, 67, 68, 69, 68, 67, 66],
         [72, 71, 70, 71, 72, 73, 74, 73, 72, 71],
         [67, 66, 65, 66, 67, 68, 69, 68, 67, 66],
         [62, 61, 60, 61, 62, 63, 64, 63, 62, 61],
         [57, 56, 55, 56, 57, 58, 59, 58, 57, 56]]]])
>>> [tile for tile in make_tile_indexes(image_shape, tile_shape)]
[(slice(None, None, None), slice(None, None, None), slice(0, 2, None), slice(0, 2, None)), (slice(None, None, None), slice(None, None, None), slice(0, 2, None), slice(1, 3, None)), (slice(None, None, None), slice(None, None, None), slice(0, 2, None), slice(2, 4, None)), (slice(None, None, None), slice(None, None, None), slice(0, 2, None), slice(3, 5, None)), (slice(None, None, None), slice(None, None, None), slice(0, 2, None), slice(4, 6, None)), (slice(None, None, None), slice(None, None, None), slice(1, 3, None), slice(0, 2, None)), (slice(None, None, None), slice(None, None, None), slice(1, 3, None), slice(1, 3, None)), (slice(None, None, None), slice(None, None, None), slice(1, 3, None), slice(2, 4, None)), (slice(None, None, None), slice(None, None, None), slice(1, 3, None), slice(3, 5, None)), (slice(None, None, None), slice(None, None, None), slice(1, 3, None), slice(4, 6, None)), (slice(None, None, None), slice(None, None, None), slice(2, 4, None), slice(0, 2, None)), (slice(None, None, None), slice(None, None, None), slice(2, 4, None), slice(1, 3, None)), (slice(None, None, None), slice(None, None, None), slice(2, 4, None), slice(2, 4, None)), (slice(None, None, None), slice(None, None, None), slice(2, 4, None), slice(3, 5, None)), (slice(None, None, None), slice(None, None, None), slice(2, 4, None), slice(4, 6, None)), (slice(None, None, None), slice(None, None, None), slice(3, 5, None), slice(0, 2, None)), (slice(None, None, None), slice(None, None, None), slice(3, 5, None), slice(1, 3, None)), (slice(None, None, None), slice(None, None, None), slice(3, 5, None), slice(2, 4, None)), (slice(None, None, None), slice(None, None, None), slice(3, 5, None), slice(3, 5, None)), (slice(None, None, None), slice(None, None, None), slice(3, 5, None), slice(4, 6, None))]
>>> [tile for tile in make_tile_indexes(image_shape, tile_shape_4x4)]
[(slice(None, None, None), slice(None, None, None), slice(0, 4, None), slice(0, 4, None)), (slice(None, None, None), slice(None, None, None), slice(0, 4, None), slice(2, 6, None)), (slice(None, None, None), slice(None, None, None), slice(0, 4, None), slice(4, 8, None)), (slice(None, None, None), slice(None, None, None), slice(2, 6, None), slice(0, 4, None)), (slice(None, None, None), slice(None, None, None), slice(2, 6, None), slice(2, 6, None)), (slice(None, None, None), slice(None, None, None), slice(2, 6, None), slice(4, 8, None)), (slice(None, None, None), slice(None, None, None), slice(4, 8, None), slice(0, 4, None)), (slice(None, None, None), slice(None, None, None), slice(4, 8, None), slice(2, 6, None)), (slice(None, None, None), slice(None, None, None), slice(4, 8, None), slice(4, 8, None))]
>>> [tile for tile in make_tiles(np.arange(3*4).reshape(1, 3, 2, 2), tile_shape)]
[array([[[[ 3,  2],
         [ 1,  0]],
<BLANKLINE>
        [[ 7,  6],
         [ 5,  4]],
<BLANKLINE>
        [[11, 10],
         [ 9,  8]]]]), array([[[[ 2,  3],
         [ 0,  1]],
<BLANKLINE>
        [[ 6,  7],
         [ 4,  5]],
<BLANKLINE>
        [[10, 11],
         [ 8,  9]]]]), array([[[[ 3,  2],
         [ 1,  0]],
<BLANKLINE>
        [[ 7,  6],
         [ 5,  4]],
<BLANKLINE>
        [[11, 10],
         [ 9,  8]]]]), array([[[[ 1,  0],
         [ 3,  2]],
<BLANKLINE>
        [[ 5,  4],
         [ 7,  6]],
<BLANKLINE>
        [[ 9,  8],
         [11, 10]]]]), array([[[[ 0,  1],
         [ 2,  3]],
<BLANKLINE>
        [[ 4,  5],
         [ 6,  7]],
<BLANKLINE>
        [[ 8,  9],
         [10, 11]]]]), array([[[[ 1,  0],
         [ 3,  2]],
<BLANKLINE>
        [[ 5,  4],
         [ 7,  6]],
<BLANKLINE>
        [[ 9,  8],
         [11, 10]]]]), array([[[[ 3,  2],
         [ 1,  0]],
<BLANKLINE>
        [[ 7,  6],
         [ 5,  4]],
<BLANKLINE>
        [[11, 10],
         [ 9,  8]]]]), array([[[[ 2,  3],
         [ 0,  1]],
<BLANKLINE>
        [[ 6,  7],
         [ 4,  5]],
<BLANKLINE>
        [[10, 11],
         [ 8,  9]]]]), array([[[[ 3,  2],
         [ 1,  0]],
<BLANKLINE>
        [[ 7,  6],
         [ 5,  4]],
<BLANKLINE>
        [[11, 10],
         [ 9,  8]]]])]
>>> canvas_shape(image_shape, (1, 3, 100, 100))
(1, 3, 150, 150)
>>> [tile for tile in make_tile_indexes(image_shape, (1, 3, 100, 100))]
[(slice(None, None, None), slice(None, None, None), slice(0, 100, None), slice(0, 100, None)), (slice(None, None, None), slice(None, None, None), slice(0, 100, None), slice(50, 150, None)), (slice(None, None, None), slice(None, None, None), slice(50, 150, None), slice(0, 100, None)), (slice(None, None, None), slice(None, None, None), slice(50, 150, None), slice(50, 150, None))]
"""

import numpy as np

# given an image shape and a tile shape, compute the resulting canvas shape, with enough padding so that a fixed-size
# tile will cover every original pixel
def canvas_shape(image_shape, tile_shape):
    rows = image_shape[2] + tile_shape[2] + (tile_shape[2] // 2 - image_shape[2]) % (tile_shape[2] // 2)
    cols = image_shape[3] + tile_shape[3] + (tile_shape[3] // 2 - image_shape[3]) % (tile_shape[3] // 2)

    return (
        image_shape[0],
        image_shape[1],
        rows,
        cols
    )


# create a mirror padding image, given an image, left, top, right, bottom padding
def pad_image(image, pad_width):
    return np.pad(image, pad_width, 'reflect')

def pad_width_for_tiling(image_shape, tile_shape):
    """
    >>> pad_width_for_tiling((1, 3, 3, 4), (1, 3, 2, 2))
    ((0, 0), (0, 0), (1, 1), (1, 1))
    >>> pad_width_for_tiling((1, 3, 3, 4), (1, 3, 4, 4))
    ((0, 0), (0, 0), (2, 3), (2, 2))
    >>> pad_width_for_tiling((1, 3, 3, 5), (1, 3, 4, 4))
    ((0, 0), (0, 0), (2, 3), (2, 3))
    >>> pad_width_for_tiling((1, 3, 5, 5), (1, 3, 4, 4))
    ((0, 0), (0, 0), (2, 3), (2, 3))
    """
    # if image_shape[2] <= tile_shape[2]:
    #     rows_pad = (0, (tile_shape[2] - image_shape[2]))
    # else:
    #     rows_pad = (tile_shape[2] // 2, tile_shape[2] // 2 + (tile_shape[2] // 2 - image_shape[2]) % (tile_shape[2] // 2))
    #
    # if image_shape[3] <= tile_shape[3]:
    #     cols_pad = (0, (tile_shape[3] - image_shape[3]))
    # else:
    #     cols_pad = (tile_shape[3] // 2, tile_shape[3] // 2 + (tile_shape[3] // 2 - image_shape[3]) % (tile_shape[3] // 2))
    rows_pad = (tile_shape[2] // 2, tile_shape[2] // 2 + (tile_shape[2] // 2 - image_shape[2]) % (tile_shape[2] // 2))
    cols_pad = (tile_shape[3] // 2, tile_shape[3] // 2 + (tile_shape[3] // 2 - image_shape[3]) % (tile_shape[3] // 2))

    return (
        (0, 0),
        (0, 0),
        rows_pad,
        cols_pad
    )


def pad_image_for_tiling(image, tile_shape):
    pad_width = pad_width_for_tiling(image.shape, tile_shape)
    return pad_image(image, pad_width)

# create tiles
def make_tile_indexes(image_shape, tile_shape):
    cs = canvas_shape(image_shape, tile_shape)
    for x in make_tile_indexes_from_canvas(cs, tile_shape):
        yield x

def make_tile_indexes_from_canvas(cs, tile_shape, offset=(0, 0)):
    # loop over rows
    cur_row = offset[0]
    while cur_row < cs[2] - tile_shape[2] // 2:
        # loop over columns
        cur_col = offset[1]
        while cur_col < cs[3] - tile_shape[3] // 2:
            yield (slice(None, None, None), slice(None, None, None), slice(cur_row, cur_row + tile_shape[2]), slice(cur_col, cur_col + tile_shape[3]))
            cur_col += tile_shape[3] // 2

        cur_row += tile_shape[2] // 2

def make_tiles(image, tile_shape):
    canvas = pad_image_for_tiling(image, tile_shape)
    for tile_ix in make_tile_indexes(image.shape, tile_shape):
        yield canvas[tile_ix]


def make_tiles_from_canvas(canvas, tile_shape, offset=(0, 0)):
    for tile_ix in make_tile_indexes_from_canvas(canvas.shape, tile_shape, offset):
        yield canvas[tile_ix]


def make_img_idx_from_canvas(image_shape, tile_shape):
    pad_width = pad_width_for_tiling(image_shape, tile_shape)
    return (slice(None), slice(None), slice(pad_width[2][0], image_shape[2] + pad_width[2][0]), slice(pad_width[3][0], image_shape[3] + pad_width[3][0]))


if __name__ == '__main__':
    import doctest

    result = doctest.testmod()
    if result.failed > 0:
        print "Failed:", result
        import sys

        sys.exit(1)
    else:
        print "Success:", result


