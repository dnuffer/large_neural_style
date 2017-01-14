import theano.tensor as T
import theano

def print_loss(loss, message, do_print):
    if do_print:
        loss = theano.printing.Print(message)(loss)
    return loss

def gram_matrix(x):
    flat = x.flatten(ndim=3)
    gram = T.tensordot(flat, flat, axes=([2], [2]))
    return gram


def content_loss(original_f, generated_f, do_print):
    # original from paper
    #loss = T.square(generated_f - original_f).sum()
    # use mean so that the loss is resolution independent
    loss = T.square(generated_f - original_f).mean()
    return print_loss(loss, 'content loss:', do_print)


def layer_style_loss(original_f, generated_f, name, do_print):
    original_gm = gram_matrix(original_f)
    generated_gm = gram_matrix(generated_f)

    N = original_f.shape[1]
    M = original_f.shape[2] * original_f.shape[3]

    loss = ((generated_gm - original_gm) ** 2).sum() / (N * M)
    return print_loss(loss, name + ' loss:', do_print)


def total_variation_loss(x, do_print):
    # use mean so that the loss is resolution independent
    loss = ((T.square(x[:, :, :-1, :-1] - x[:, :, 1:, :-1]) + T.square(x[:, :, :-1, :-1] - x[:, :, :-1, 1:])) ** 1.25).mean()
    return print_loss(loss, 'tv loss:', do_print)


def total_loss(photo_features, art_features, gen_features, generated_image, content_layer_names, style_layer_names,
               content_weight, style_weight, style_layer_weight_factor, total_variation_weight, do_print):
    losses = []

    # content loss
    for content_layer_name in content_layer_names:
        losses.append(print_loss(content_weight / len(content_layer_names) * content_loss(photo_features[content_layer_name],
                                                                               gen_features[content_layer_name], do_print), 'weighted content loss: ', do_print))

    # style loss
    for i, style_layer_name in enumerate(style_layer_names):
        losses.append(print_loss(style_weight * (style_layer_weight_factor ** i) / (sum([style_layer_weight_factor ** i for i in range(len(style_layer_names))]))  * layer_style_loss(art_features[style_layer_name],
                                                                               gen_features[style_layer_name],
                                                                               style_layer_name, do_print), 'weighted style ' + style_layer_name + ' loss:', do_print))

    # total variation penalty
    losses.append(print_loss(total_variation_weight * total_variation_loss(generated_image, do_print), 'weighted tv loss: ', do_print))

    return print_loss(T.sum(losses), 'total loss: ', do_print)
