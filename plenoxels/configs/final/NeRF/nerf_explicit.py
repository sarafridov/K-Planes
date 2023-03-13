config = {
 'expname': 'lego_explicit',
 'logdir': './logs/syntheticstatic',
 'device': 'cuda:0',

 'data_downsample': 1.0,
 'data_dirs': ['data/nerf_synthetic/lego'],
 'contract': False,
 'ndc': False,

 # Optimization settings
 'num_steps': 30001,
 'batch_size': 4096,
 'optim_type': 'adam',
 'scheduler_type': 'warmup_cosine',
 'lr': 0.01,

 # Regularization
 'plane_tv_weight': 0.0001,
 'plane_tv_weight_proposal_net': 0.0001,
 'histogram_loss_weight': 1.0,
 'distortion_loss_weight': 0.001,

 # Training settings
 'save_every': 30000,
 'valid_every': 30000,
 'save_outputs': True,
 'train_fp16': True,

 # Raymarching settings
 'single_jitter': False,
 'num_samples': 48,
 # proposal sampling
 'num_proposal_samples': [256, 128],
 'num_proposal_iterations': 2,
 'use_same_proposal_network': False,
 'use_proposal_weight_anneal': True,
 'proposal_net_args_list': [
   {'num_input_coords': 3, 'num_output_coords': 8, 'resolution': [64, 64, 64]},
   {'num_input_coords': 3, 'num_output_coords': 8, 'resolution': [128, 128, 128]}
 ],

 # Model settings
 'multiscale_res': [1, 2, 4],
 'density_activation': 'trunc_exp',
 'concat_features_across_scales': True,
 'linear_decoder': True,
 'linear_decoder_layers': 4,
 'grid_config': [{
  'grid_dimensions': 2,
  'input_coordinate_dim': 3,
  'output_coordinate_dim': 32,
  'resolution': [128, 128, 128]
 }],
}
