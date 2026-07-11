import pytest
import numpy as np
import torch
from scipy import fftpack
from digit_sdk.digit_reconstruct import _dct2, _idct2


@pytest.mark.timeout(10)
def test_dct_equivalence():
    """Verify torch _dct2 matches scipy.fftpack.dctn with norm='ortho'."""
    x = np.random.randn(32, 32).astype(np.float32)
    s = fftpack.dctn(x, norm='ortho')
    t = _dct2(torch.from_numpy(x)).numpy()
    mse = ((s - t)**2).mean()
    assert mse < 1e-5, f'DCT MSE: {mse:.2e}'


@pytest.mark.timeout(10)
def test_idct_equivalence():
    """Verify torch _idct2 matches scipy.fftpack.idctn with norm='ortho'."""
    x = np.random.randn(32, 32).astype(np.float32)
    s = fftpack.idctn(x, norm='ortho')
    t = _idct2(torch.from_numpy(x)).numpy()
    mse = ((s - t)**2).mean()
    assert mse < 1e-5, f'IDCT MSE: {mse:.2e}'


@pytest.mark.timeout(10)
def test_dct_idct_roundtrip():
    """Verify _idct2(_dct2(x)) == x to floating-point precision."""
    x = np.random.randn(32, 32).astype(np.float32)
    xt = torch.from_numpy(x)
    reconstructed = _idct2(_dct2(xt)).numpy()
    mse = ((x - reconstructed)**2).mean()
    assert mse < 1e-5, f'Roundtrip MSE: {mse:.2e}'
