from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

class CommonConfig:
    k_n = 16
    num_points = 40960
    batch_size = 4
    val_batch_size = 3
    train_steps = 500
    val_steps = 100
    d_state = 16
    d_conv = 4
    expand = 2
    cbl_weight = 0.1
    cbl_temperature = 0.5
    cbl_stages = (2, 3)
    mix3d_prob = 0.3
    mix3d_gap = 2.0
    normal_k = 20
    geo_feature_dim = 5
    stage_points = [2560, 5120, 20480, 40960]
    stage_dims = [160, 160, 176, 176]
    stage_blocks = [3, 3, 4, 4]
    stage_weights = [0.1, 0.2, 0.3, 0.4]
    gadpe_num_freqs = 4
    noise_init = 4
    max_epoch = 200
    learning_rate = 0.0002
    weight_decay = 0.0005
    warmup_epochs = 5
    eta_min = 1e-06
    ema_decay = 0.999
    boundary_weight = 2.0
    saving = True
    saving_path = None
    ignored_label_inds = []

class BFConfig(CommonConfig):
    dataset = 'bf'
    experiment_name = 'CascadeMambaNet_BF'
    data_root = ROOT / 'data' / 'BF'
    num_classes = 8
    sub_grid_size = 0.04
    use_color = True
    stage_grids = [0.16, 0.12, 0.06, 0.04]
    grid_size_serial = 0.02
    lovasz_weight = 0.5
    results_root = ROOT / 'results'
    logs_root = ROOT / 'logs'

class ZAHAConfig(CommonConfig):
    dataset = 'zaha'
    experiment_name = 'CascadeMambaNet_ZAHA'
    data_root = ROOT / 'data' / 'ZAHA'
    num_classes = 15
    sub_grid_size = 0.2
    use_color = False
    stage_grids = [0.8, 0.6, 0.3, 0.2]
    grid_size_serial = 0.1
    lovasz_weight = 0.7
    processed_data_root = data_root / 'processed' / 'grid_0.200'
    results_root = ROOT / 'results'
    logs_root = ROOT / 'logs'
    sampling_state_root = results_root / 'sampling_state'
    max_cached_clouds = 4
    class_weight_smoothing = 0.02
    class_weight_max = 10.0
    periodic_checkpoint_start_epoch = 20
    periodic_checkpoint_interval = 5
DATASET_CONFIGS = {'bf': BFConfig, 'zaha': ZAHAConfig}
