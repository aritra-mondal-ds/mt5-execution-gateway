def test_torch_cuda():
    import torch
    assert torch.cuda.is_available(), "CUDA not available"
    assert "3060" in torch.cuda.get_device_name(0)

def test_mt5_import():
    import MetaTrader5 as mt5
    assert mt5 is not None

def test_core_imports():
    import pandas, polars, lightgbm, ccxt, loguru
    assert all([pandas, polars, lightgbm, ccxt, loguru])