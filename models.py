from tensorflow.keras import layers, models

from tensorflow.python.keras.engine.keras_tensor import KerasTensor

from construct_model import create_u_encoder, upsample,eca_block




class U2Model(models.Model):
    def __init__(self, padding: str ='same', build: bool = True, **kwargs):
        super(U2Model, self).__init__()

        self.padding = padding
        self.sleep_epoch_length = kwargs['sleep_epoch_len']
        self.sequence_length = kwargs['preprocess']['sequence_epochs']
        self.filters = kwargs['train']['filters']
        self.kernel_size = kwargs['train']['kernel_size']
        self.pooling_sizes = kwargs['train']['pooling_sizes']
        self.activation = kwargs['train']['activation']
        self.u_depths = kwargs['train']['u_depths']
        self.u_inner_filter = kwargs['train']['u_inner_filter']


        if build:
            super().__init__(*self.init_model())

    def init_model(self, input: KerasTensor = None) -> (list, list):
        if input is None:
            input = layers.Input(shape=(self.sequence_length * self.sleep_epoch_length, 1, 1))

        l_name = "single_model_enc"

        # encoder 1
        u1 = create_u_encoder(input, self.filters[0], self.kernel_size, self.pooling_sizes[0],
                              middle_layer_filter=self.u_inner_filter, depth=self.u_depths[0],
                              pre_name=l_name, idx=1, padding=self.padding, activation=self.activation)
        u1 = layers.Conv2D(int(u1.get_shape()[-1] * 0.5), (1, 1),
                           name=f"{l_name}_reduce_dim_layer_1",
                           padding=self.padding, activation=self.activation)(u1)
        pool = layers.MaxPooling2D((self.pooling_sizes[0], 1), name=f"{l_name}_pool1")(u1)

        # encoder 2
        u2 = create_u_encoder(pool, self.filters[1], self.kernel_size, self.pooling_sizes[1],
                              middle_layer_filter=self.u_inner_filter, depth=self.u_depths[1],
                              pre_name=l_name, idx=2, padding=self.padding, activation=self.activation)
        u2 = layers.Conv2D(int(u2.get_shape()[-1] * 0.5), (1, 1),
                           name=f"{l_name}_reduce_dim_layer_2",
                           padding=self.padding, activation=self.activation)(u2)
        pool = layers.MaxPooling2D((self.pooling_sizes[1], 1), name=f"{l_name}_pool2")(u2)

        # encoder 3
        u3 = create_u_encoder(pool, self.filters[2], self.kernel_size, self.pooling_sizes[2],
                              middle_layer_filter=self.u_inner_filter, depth=self.u_depths[2],
                              pre_name=l_name, idx=3, padding=self.padding, activation=self.activation)
        u3 = layers.Conv2D(int(u3.get_shape()[-1] * 0.5), (1, 1),
                           name=f"{l_name}_reduce_dim_layer_3",
                           padding=self.padding, activation=self.activation)(u3)
        pool = layers.MaxPooling2D((self.pooling_sizes[2], 1), name=f"{l_name}_pool3")(u3)

        # encoder 4
        u4 = create_u_encoder(pool, self.filters[3], self.kernel_size, self.pooling_sizes[3],
                              middle_layer_filter=self.u_inner_filter, depth=self.u_depths[3],
                              pre_name=l_name, idx=4, padding=self.padding, activation=self.activation)
        u4 = layers.Conv2D(int(u4.get_shape()[-1] * 0.5), (1, 1),
                           name=f"{l_name}_reduce_dim_layer_4",
                           padding=self.padding, activation=self.activation)(u4)
        pool = layers.MaxPooling2D((self.pooling_sizes[3], 1), name=f"{l_name}_pool4")(u4)

        # encoder 5
        u5 = create_u_encoder(pool, self.filters[4], self.kernel_size, self.pooling_sizes[3],
                              middle_layer_filter=self.u_inner_filter, depth=self.u_depths[3],
                              pre_name=l_name, idx=5, padding=self.padding, activation=self.activation)
        u5 = layers.Conv2D(int(u5.get_shape()[-1] * 0.5), (1, 1),
                           name=f"{l_name}_reduce_dim_layer_5",
                           padding=self.padding, activation=self.activation)(u5)

        l_name = "single_model_dec"

        up4 = upsample(u4, pre_name=l_name, idx=4)(u5)

        d4 = create_u_encoder(layers.concatenate([up4, u4], axis=-1), self.filters[3], kernel_size=self.kernel_size,
                              pooling_size=self.pooling_sizes[3], middle_layer_filter=self.u_inner_filter,
                              depth=self.u_depths[3], pre_name=l_name, idx=4, padding=self.padding,
                              activation=self.activation)
        d4 = layers.Conv2D(int(d4.get_shape()[-1] * 0.5), (1, 1),
                           name=f"{l_name}_reduce_dim_4",
                           padding=self.padding, activation=self.activation)(d4)

        # decoder 3
        up3 = upsample(u3, pre_name=l_name, idx=3)(d4)

        d3 = create_u_encoder(layers.concatenate([up3, u3], axis=-1), self.filters[2], kernel_size=self.kernel_size,
                              pooling_size=self.pooling_sizes[2], middle_layer_filter=self.u_inner_filter,
                              depth=self.u_depths[2], pre_name=l_name, idx=3, padding=self.padding,
                              activation=self.activation)
        d3 = layers.Conv2D(int(d3.get_shape()[-1] * 0.5), (1, 1),
                           name=f"{l_name}_reduce_dim_3",
                           padding=self.padding, activation=self.activation)(d3)

        # decoder 2
        up2 = upsample(u2, pre_name=l_name, idx=2)(d3)

        d2 = create_u_encoder(layers.concatenate([up2, u2], axis=-1), self.filters[1], kernel_size=self.kernel_size,
                              pooling_size=self.pooling_sizes[1], middle_layer_filter=self.u_inner_filter,
                              depth=self.u_depths[1], pre_name=l_name, idx=2, padding=self.padding,
                              activation=self.activation)
        d2 = layers.Conv2D(int(d2.get_shape()[-1] * 0.5), (1, 1),
                           name=f"{l_name}_reduce_dim_2",
                           padding=self.padding, activation=self.activation)(d2)

        # decoder 1
        up1 = upsample(u1, pre_name=l_name, idx=1)(d2)

        d1 = create_u_encoder(layers.concatenate([up1, u1], axis=-1), self.filters[0], kernel_size=self.kernel_size,
                              pooling_size=self.pooling_sizes[0], middle_layer_filter=self.u_inner_filter,
                              depth=self.u_depths[0], pre_name=l_name, idx=1, padding=self.padding,
                              activation=self.activation)

        zpad = layers.ZeroPadding2D(padding=((
            int((self.sequence_length * self.sleep_epoch_length - d1.shape[1]) // 2)),
            0))(d1)
        att = eca_block(zpad)
        reshape = layers.Reshape((self.sequence_length, self.sleep_epoch_length, self.filters[0]))(att)
        reshape = layers.Conv2D(self.filters[0], (1, 1), activation='tanh', padding='same')(reshape)
        pool = layers.AveragePooling2D((1, self.sleep_epoch_length))(reshape)
        out = layers.Conv2D(5, (self.kernel_size, 1), padding=self.padding, activation='softmax')(pool)

        return [input], [out]




if __name__ == '__main__':
    import yaml
    with open("hyperparameters.yaml", encoding='utf-8') as f:
        hyp = yaml.full_load(f)
    model: models.Model = U2Model(**hyp)
    model.summary()


