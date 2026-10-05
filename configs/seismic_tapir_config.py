# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""First-pass TAPIR configuration for procedural seismic training."""

from jaxline import base_config
from ml_collections import config_dict


def get_config(config_string: str | None = None) -> config_dict.ConfigDict:
  """Returns a conservative single-device synthetic training config."""
  if config_string not in (None, '', 'smoke'):
    raise ValueError(
        f'Unknown config variant {config_string!r}; expected "smoke" or none.'
    )
  smoke = config_string == 'smoke'
  config = base_config.get_base_config()
  config.training_steps = 1 if smoke else 20000
  config.shared_module_names = ('tapir_model',)
  config.dataset_names = ('seismic',)
  config.eval_modes = ('eval_seismic_synthetic',)
  config.checkpoint_dir = (
      './checkpoints/seismic_tapir_smoke/'
      if smoke
      else './checkpoints/seismic_tapir/'
  )
  config.evaluate_every = 100 if smoke else 1000

  synthetic_geometry = dict(
      num_frames=2 if smoke else 24,
      height=64 if smoke else 256,
      width=64 if smoke else 256,
      num_horizons=4 if smoke else 16,
      num_queries=2 if smoke else 64,
      wavelet_length=17 if smoke else 33,
      min_wavelet_frequency=0.06,
      max_wavelet_frequency=0.14,
      noise_std=0.12,
      max_fault_throw=2.0 if smoke else 12.0,
      fault_probability=0.7,
      termination_probability=0.25,
      reverse_probability=0.5,
  )

  config.experiment_kwargs = config_dict.ConfigDict(
      dict(
          config=dict(
              sweep_name='synthetic_seismic_first_pass',
              save_final_checkpoint_as_npy=True,
              optimizer=dict(
                  base_lr=1e-4,
                  max_norm=1.0,
                  weight_decay=1e-2,
                  schedule_type='cosine',
                  cosine_decay_kwargs=dict(
                      init_value=0.0,
                  warmup_steps=0 if smoke else 500,
                      end_value=1e-6,
                  ),
                  optimizer='adam',
                  adam_kwargs=dict(b1=0.9, b2=0.95, eps=1e-8),
              ),
              fast_variables=tuple(),
              shared_modules=dict(
                  shared_module_names=config.get_oneway_ref(
                      'shared_module_names'
                  ),
                  tapir_model_kwargs=dict(
                      bilinear_interp_with_depthwise_conv=False,
                      pyramid_level=0,
                      use_causal_conv=False,
                      initial_resolution=(64, 64) if smoke else (256, 256),
                  ),
              ),
              datasets=dict(
                  dataset_names=config.get_oneway_ref('dataset_names'),
                  seismic_kwargs=dict(
                      batch_dims=1,
                      seed=0,
                      **synthetic_geometry,
                  ),
                  seismic_eval_kwargs=dict(
                      num_samples=2 if smoke else 16,
                      seed=10000,
                      **synthetic_geometry,
                  ),
              ),
              supervised_point_prediction_kwargs=dict(
                  input_key='seismic',
                  prediction_algo='cost_volume_regressor',
                  model_key='tapir_model',
                  loss_type='seismic',
                  position_loss_weight=0.05,
                  depth_loss_weight=1.0,
                  lateral_loss_weight=0.25,
                  huber_loss_delta=2.0,
                  expected_dist_thresh=2.0,
                  train_chunk_size=2 if smoke else 16,
                  eval_chunk_size=2 if smoke else 16,
                  eval_inference_resolution=(64, 64)
                  if smoke
                  else (256, 256),
                  eval_metrics_resolution=(64, 64)
                  if smoke
                  else (256, 256),
              ),
              checkpoint_dir=config.get_oneway_ref('checkpoint_dir'),
              evaluate_every=config.get_oneway_ref('evaluate_every'),
              eval_modes=config.get_oneway_ref('eval_modes'),
              training=dict(
                  n_training_steps=config.get_oneway_ref('training_steps')
              ),
              inference=dict(
                  input_video_path='',
                  output_video_path='',
                  resize_height=256,
                  resize_width=256,
                  num_points=20,
              ),
          )
      )
  )

  config.train_checkpoint_all_hosts = False
  config.save_checkpoint_interval = 1 if smoke else 100
  config.eval_initial_weights = not smoke
  config.one_off_evaluate = True
  config.lock()
  return config
