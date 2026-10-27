import subprocess
import sys


def test_torch_cuda():
    code = (
        "import torch; "
        "assert torch.cuda.is_available(), 'CUDA not available'; "
        "assert '3060' in torch.cuda.get_device_name(0), f'Wrong GPU: {torch.cuda.get_device_name(0)}'; "
        "print('CUDA OK')"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert res.returncode == 0, f"CUDA check failed: {res.stderr}"


def test_mt5_import():
    import MetaTrader5 as mt5
    assert mt5 is not None


def test_core_imports():
    import pandas, polars, lightgbm, ccxt, loguru
    assert all([pandas, polars, lightgbm, ccxt, loguru])
