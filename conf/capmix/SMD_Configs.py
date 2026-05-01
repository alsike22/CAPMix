class Config(object):
    def __init__(self):
        # datasets
        self.dataset = 'SMD'
        # model configs
        self.input_channels = 38
        self.kernel_size = 8
        self.stride = 1
        self.final_out_channels = 32

        self.dropout = 0.21773924181332105
        self.features_len = 6
        self.window_size = 32
        self.time_step = 16

        # training configs
        self.num_epoch = 25

        # optimizer parameters
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 1.2941992302742048e-05
        self.weight = 5e-3

        # data parameters
        self.drop_last = False
        self.batch_size = 512
        # trend rate
        self.trend_rate = 0.01
        # negative sample rates
        self.rate = 0.6419184094618225
        # number of trend dimensions
        self.dim = 5
        # minimum cut length
        self.cut_rate = 24

        # Anomaly quantile of fixed threshold
        self.detect_nu = 0.001
        # Methods for determining thresholds ("direct","fix","floating","one-anomaly")
        self.threshold_determine = 'floating'

        ## For Mixup >0 mixup, == 0 no mixup
        self.gamma = 2.2535073483432004
        self.alpha =  0.6087822690109689
        self.layer_mix = 3


