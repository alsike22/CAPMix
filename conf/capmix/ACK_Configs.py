class Config(object):
    def __init__(self):
        # datasets
        self.dataset = 'ACK'
        # model configs
        self.input_channels = 31
        self.kernel_size = 8
        self.stride = 1
        self.final_out_channels = 32
        self.dropout = 0.4889719173294546
        self.features_len = 6
        self.window_size = 32
        self.time_step = 16

        # training configs
        self.num_epoch = 70

        # optimizer parameters
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 0.0013507300490137012
        self.weight = 0.005

        # data parameters
        self.drop_last = False
        self.batch_size = 512

        # trend rate
        self.trend_rate = 0.96

        # negative sample rates
        self.rate = 0.973518065808672

        # number of trend dimensions
        self.dim = 5

        # minimum cut length
        self.cut_rate = 4

        # Anomaly quantile of fixed threshold
        self.detect_nu = 0.001

        # Methods for determining thresholds ("direct","fix","floating","one-anomaly")
        self.threshold_determine = 'floating'


        ## For Mixup >0 mixup, == 0 no mixup
        self.gamma = 2.3625955082957835
        self.alpha = 0.8414357140122652
        self.layer_mix = 1