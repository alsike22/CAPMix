class Config(object):
    def __init__(self):
        # datasets
        self.dataset = 'ASD'
        # model configs
        self.input_channels = 19
        self.kernel_size = 8
        self.stride = 1
        self.final_out_channels = 32

        self.dropout = 0.48638879714797634
        self.features_len = 6
        self.window_size = 32
        self.time_step = 16

        # training configs
        self.num_epoch = 100

        # optimizer parameters
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 0.0009504440850827242
        self.weight = 5e-3

        # data parameters
        self.drop_last = False
        self.batch_size = 512
        # trend rate
        self.trend_rate = 0.060000000000000005
        # negative sample rates
        self.rate = 0.9457777400303025
        # number of trend dimensions
        self.dim = 5
        # minimum cut length
        self.cut_rate = 32

        # Anomaly quantile of fixed threshold
        self.detect_nu = 0.001
        # Methods for determining thresholds ("direct","fix","floating","one-anomaly")
        self.threshold_determine = 'floating'

        ## For Mixup >0 mixup, == 0 no mixup
        self.gamma = 2.625240342618447
        self.alpha =  0.4609242208851774
        self.layer_mix = 0

