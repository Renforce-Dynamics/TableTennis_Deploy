# Setup with uv

This project now supports `uv` for fast Python environment management.

## Quick Start

### Option 1: Automated Setup (Recommended)
```bash
./setup.sh
```

### Option 2: Manual Setup

1. **Install uv** (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. **Install dependencies**:
```bash
uv sync
```

3. **Install unitree_sdk2_python**:
```bash
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python
uv pip install -e .
cd ..
```

## Usage

### Run with uv (no activation needed)
```bash
uv run python deploy_mujoco/deploy_mujoco.py
uv run python test_table_tennis.py
```

### Or activate the virtual environment
```bash
source .venv/bin/activate
python deploy_mujoco/deploy_mujoco.py
```

## Benefits of uv

- ⚡ **10-100x faster** than pip
- 🔒 **Reproducible builds** with uv.lock
- 🎯 **No need for conda** - uv handles everything
- 📦 **Better dependency resolution**

## Installing Unitree SDK (for real robot deployment)

The unitree SDK requires cyclonedds C library:

```bash
# Install cyclonedds system dependency
sudo apt-get install -y cyclonedds

# Then install the Python SDK
cd unitree_sdk2_python
uv pip install -e .
cd ..
```

For MuJoCo simulation only, unitree SDK is not required.

## Troubleshooting

If you encounter issues with PyTorch CUDA, ensure you have CUDA 12.1 installed, or modify the PyTorch index in `pyproject.toml` to match your CUDA version.

**Note:** If you have a conda environment active, you may need to deactivate it or run uv commands with `unset VIRTUAL_ENV` prefix.
