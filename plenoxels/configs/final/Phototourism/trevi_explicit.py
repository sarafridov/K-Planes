config = {
 'expname': 'trevi_explicit',
 'logdir': './logs/trevi',
 'device': 'cuda:0',

 'data_downsample': 1,
 'data_dirs': ['data/phototourism/trevi-fountain'],
 'contract': True,
 'ndc': False,
 'scene_bbox': [[-2.0, -2.0, -2.0], [2.0, 2.0, 2.0]],
 'global_scale': [5, 6, 4],
 'global_translation': [0, 0, -1],

 # Optimization settings
 'num_steps': 30001,
 'batch_size': 4096,
 'optim_type': 'adam',
 'scheduler_type': 'warmup_cosine',
 'lr': 0.01,
 'app_optim_lr': 0.1,
 'app_optim_n_epochs': 10,

 # Regularization
 'plane_tv_weight': 0.0002,
 'plane_tv_weight_proposal_net': 0.0002,
 'distortion_loss_weight': 0.0,
 'histogram_loss_weight': 1.0,

 # Training settings
 'save_every': 10000,
 'valid_every': 10000,
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
   {'num_input_coords': 3, 'num_output_coords': 8, 'resolution': [128, 128, 128]},
   {'num_input_coords': 3, 'num_output_coords': 8, 'resolution': [256, 256, 256]}
 ],

 # Model settings
 'multiscale_res': [1, 2, 4, 8],
 'density_activation': 'trunc_exp',
 'concat_features_across_scales': True,
 'linear_decoder': True,
 'linear_decoder_layers': 1,
 'appearance_embedding_dim': 32,
 'grid_config': [{
  'grid_dimensions': 2,
  'input_coordinate_dim': 3,
  'output_coordinate_dim': 32,
  'resolution': [64, 64, 64]
 }],
}
