class Config(object):
    def __init__(self):
        # datasets
        self.dataset = 'ASD'
        # model configs
        self.input_channels = 19
        self.kernel_size = 8
        self.stride = 1
        self.final_out_channels = 32

        self.dropout = 0.3781139173496069
        self.features_len = 6
        self.window_size = 32
        self.time_step = 16

        # training configs
        self.num_epoch = 30

        # optimizer parameters
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 0.001563074752429823
        self.weight = 5e-3

        # data parameters
        self.drop_last = False
        self.batch_size = 512
        # trend rate
        self.trend_rate = 0.01
        # negative sample rates
        self.rate = 1
        # number of trend dimensions
        self.dim = 5
        # minimum cut length
        self.cut_rate = 10

        # Anomaly quantile of fixed threshold
        self.detect_nu = 0.001
        # Methods for determining thresholds ("direct","fix","floating","one-anomaly")
        self.threshold_determine = 'floating'

        ## For Mixup >0 mixup, == 0 no mixup
        self.gamma = 1.5
        self.alpha = 0.529189018235973
        self.layer_mix = 1







