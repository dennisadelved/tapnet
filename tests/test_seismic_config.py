"""Optional config contract tests."""

import pytest

pytest.importorskip('jaxline')

from configs import seismic_tapir_config


def test_default_config_uses_seismic_dataset_and_loss():
  config = seismic_tapir_config.get_config()
  task = config.experiment_kwargs.config.supervised_point_prediction_kwargs
  assert config.dataset_names == ('seismic',)
  assert config.eval_modes == ('eval_seismic_synthetic',)
  assert task.input_key == 'seismic'
  assert task.loss_type == 'seismic'


def test_smoke_config_is_small_and_one_step():
  config = seismic_tapir_config.get_config('smoke')
  experiment = config.experiment_kwargs.config
  geometry = experiment.datasets.seismic_kwargs
  assert config.training_steps == 1
  assert geometry.num_frames == 2
  assert geometry.height == 64
  assert experiment.shared_modules.tapir_model_kwargs.initial_resolution == (
      64,
      64,
  )
  assert config.one_off_evaluate
