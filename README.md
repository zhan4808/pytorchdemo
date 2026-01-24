# PyTorch Compile Demo

This repo contains three small scripts to demonstrate `torch.compile`,
FX graph inspection, and a custom backend prototype.

## Environment (conda)

```bash
conda create -n accel_project python=3.11 -y
conda activate accel_project
pip install torch torchvision
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## Run the demos

```bash
python test_compile_basics.py
python explore_fx_graph.py
python my_accel_backend_prototype.py
```

Notes:
- The scripts are device-agnostic by default.
- On Apple Silicon, `torch.compile` runs on CPU; MPS is not supported by Inductor yet.
- `my_accel_backend_prototype.py` now routes ops through `kernel_lib.py` (Python
  placeholder kernels).

## Expected output (high level)

`test_compile_basics.py`
- Example 1 prints an output shape like `torch.Size([128, 128])`
- Example 2 prints a graph summary (node list + tabular + generated code), then an output shape like `torch.Size([32, 64])`
- Example 3 prints a message about Inductor output and a numeric output sum

`explore_fx_graph.py`
- Prints the FX graph table
- Prints each node with `op`, `target`, `args`, `kwargs`
- Prints the generated forward code

`my_accel_backend_prototype.py`
- Prints the ops seen by the backend (supported vs fallback)
- Shows `EXECUTING ON MY_ACCEL` and per-op logs
- Prints final output shape and output sums for two calls
- Kernel calls go through `kernel_lib.py` (GEMM/ReLU/Softmax placeholders)