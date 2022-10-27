import torch.nn as nn
import torch.nn.functional as F

from model import common


def make_model(args):
    return ALEATORIC(args)


class ALEATORIC(nn.Module):
    def __init__(self, config):
        super(ALEATORIC, self).__init__()
        self.drop_rate = config.drop_rate
        in_channels = config.in_channels
        n_classes = config.n_classes
        self.depth = 3
        filter_config  = (64, 128, 256) 

        self.encoders = nn.ModuleList()
        self.decoders_mean = nn.ModuleList()
        self.decoders_var = nn.ModuleList()

        # setup number of conv-bn-relu blocks per module and number of filters
        encoder_n_layers = (2, 2, 3, 3, 3)
        encoder_filter_config = (in_channels,) + filter_config
        decoder_n_layers = (3, 3, 3, 2, 1)
        decoder_filter_config = filter_config[::-1] + (filter_config[0],)

        for i in range(self.depth):
            # encoder architecture
            self.encoders.append(_Encoder(encoder_filter_config[i],
                                          encoder_filter_config[i + 1],
                                          encoder_n_layers[i]))

            # decoder architecture
            self.decoders_mean.append(_Decoder(decoder_filter_config[i],
                                               decoder_filter_config[i + 1],
                                               decoder_n_layers[i]))

            # decoder architecture
            self.decoders_var.append(_Decoder(decoder_filter_config[i],
                                              decoder_filter_config[i + 1],
                                              decoder_n_layers[i]))

        # final classifier (equivalent to a fully connected layer)
        self.classifier_mean = nn.Conv2d(filter_config[0], n_classes, 3, 1, 1)
        self.classifier_var = nn.Conv2d(filter_config[0], n_classes, 3, 1, 1)

    def forward(self, x):
        indices = []
        unpool_sizes = []
        feat = x

        # encoder path, keep track of pooling indices and features size
        for i in range(self.depth):
            (feat, ind), size = self.encoders[i](feat)
            if i == 1:
                feat = F.dropout(feat, p=self.drop_rate)
            indices.append(ind)
            unpool_sizes.append(size)

        feat_mean = feat
        feat_var = feat
        # decoder path, upsampling with corresponding indices and size
        for i in range(self.depth):
            feat_mean = self.decoders_mean[i](feat_mean, indices[self.depth-1 - i], unpool_sizes[self.depth-1 - i])
            feat_var = self.decoders_var[i](feat_var, indices[self.depth-1 - i], unpool_sizes[self.depth-1 - i])
            if i == 0:
                feat_mean = F.dropout(feat_mean, p=self.drop_rate)
                feat_var = F.dropout(feat_var, p=self.drop_rate)

        output_mean = self.classifier_mean(feat_mean)
        output_var = self.classifier_var(feat_var)

        results = {'mean': output_mean, 'var': output_var}
        return results


class _Encoder(nn.Module):
    def __init__(self, n_in_feat, n_out_feat, n_blocks=2):
        """Encoder layer follows VGG rules + keeps pooling indices
        Args:
            n_in_feat (int): number of input features
            n_out_feat (int): number of output features
            n_blocks (int): number of conv-batch-relu block inside the encoder
            drop_rate (float): dropout rate to use
        """
        super(_Encoder, self).__init__()

        layers = [nn.Conv2d(n_in_feat, n_out_feat, 3, 1, 1),
                  nn.BatchNorm2d(n_out_feat),
                  nn.ReLU()]

        if n_blocks > 1:
            layers += [nn.Conv2d(n_out_feat, n_out_feat, 3, 1, 1),
                       nn.BatchNorm2d(n_out_feat),
                       nn.ReLU()]

        self.features = nn.Sequential(*layers)

    def forward(self, x):
        output = self.features(x)
        return F.max_pool2d(output, 2, 2, return_indices=True), output.size()


class _Decoder(nn.Module):
    """Decoder layer decodes the features by unpooling with respect to
    the pooling indices of the corresponding decoder part.
    Args:
        n_in_feat (int): number of input features
        n_out_feat (int): number of output features
        n_blocks (int): number of conv-batch-relu block inside the decoder
        drop_rate (float): dropout rate to use
    """

    def __init__(self, n_in_feat, n_out_feat, n_blocks=2):
        super(_Decoder, self).__init__()

        layers = [nn.Conv2d(n_in_feat, n_in_feat, 3, 1, 1),
                  nn.BatchNorm2d(n_in_feat),
                  nn.ReLU()]

        if n_blocks > 1:
            layers += [nn.Conv2d(n_in_feat, n_out_feat, 3, 1, 1),
                       nn.BatchNorm2d(n_out_feat),
                       nn.ReLU()]

        self.features = nn.Sequential(*layers)

    def forward(self, x, indices, size):
        unpooled = F.max_unpool2d(x, indices, 2, 2, 0, size)
        return self.features(unpooled)
