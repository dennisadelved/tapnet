"""Guards the intentionally separate PyTorch-only dependency list."""

from pathlib import Path


def test_torch_requirements_cover_direct_imports_without_selecting_torch_build():
  requirements = Path('requirements_seismic_torch.txt').read_text(
      encoding='utf-8'
  )
  packages = {
      line.strip().lower()
      for line in requirements.splitlines()
      if line.strip() and not line.lstrip().startswith('#')
  }

  assert {
      'numpy',
      'einshape',
      'dm-tree',
      'matplotlib',
      'pyzgy==0.1.1',
  } <= packages
  assert 'torch' not in packages

