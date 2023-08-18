import numpy as np


class Hyp:
    """
    A class contain model's hyperparameters
    """
    def __init__(self, gpu_num: int = 1, modal_type: int = 1):
        """
        :param gpu_num: the training gpu numbers
        :param modal_type: 0 stand for single modal while 1 stand for multi modal
        :param epochnumber: the length of training sequence
        :param biggroup_size: the size of big group
        :param window_stride: the window stride for data enhancement
        :param k_validation: validation folds
        :param fliters: the list of filter number for each U structure
        :param kernel_size: the list of kernel size for each U structure
        :param pooling_size: the list of pooling layer size of different U structure
        ::
        """
        # global
        self.gpu_num = gpu_num
        self.modal_type = modal_type

        # preprocess
        self.epochnumber = 20
        self.biggroup_size = 40
        self.window_stride = 10

        # validation
        self.k_validation = 20

        # model
        self.fliters = [16, 32, 64, 128, 256]
        self.kernel_size = 5
        self.pooling_size = [10, 8, 6, 4]
        self.activation = "relu"
        self.dilation_rate = 1  # TODO: delete this
        self.mfe_fliters = 8

        # training
        self.epochs = 60
        self.batch_size = 4

        # adjust
        self.patience = 5
        self.class_weights = np.array(
            [1.0, 1.80, 1.0, 1.25, 1.20])

    def print_hyp(self):
        print(28*'='+'total'+28*'=')
        print('gpu_num = ',self.gpu_num)
        print('modal_type = ', self.modal_type)

        #print('\n')
        print(28 * '=' + 'preprocessing' + 28 * '=')
        print('epochnumber = ',self.epochnumber)
        print('biggroup_size = ',self.biggroup_size)
        print('window_stride = ',self.window_stride)

        #print('\n')
        print(28 * '=' + 'k_validation' + 28 * '=')
        print('k_validation = ',self.k_validation)

        #print('\n')
        print(28 * '=' + 'model' + 28 * '=')
        print('flters = ', self.fliters)
        print('kernel_size = ', self.kernel_size)
        print('pooling_size = ',self.pooling_size)
        print('activation = ',self.activation)
        print('dilation_rate = ',self.dilation_rate)

        #print('\n')
        print(28 * '=' + 'training' + 28 * '=')
        print('epochs = ',self.epochs)
        print('batch_size = ',self.batch_size)

        #print('\n')
        print(28 * '=' + 'adjust' + 28 * '=')
        print('patience = ',self.patience)
        print('class_weights = ',self.class_weights)
