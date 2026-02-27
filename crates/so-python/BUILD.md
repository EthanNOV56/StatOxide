# Building StatOxide Python Bindings

## Prerequisites

### 1. Python Development Files
You need Python development headers and libraries installed:

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install python3-dev python3.11-dev
```

**CentOS/RHEL:**
```bash
sudo yum install python3-devel
```

**macOS:**
```bash
brew install python@3.11
```

### 2. Rust Toolchain
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

### 3. Maturin (Recommended)
```bash
pip install maturin
```

## Building with Maturin (Recommended)

Maturin handles Python linking automatically:

```bash
cd /path/to/statoxide/crates/so-python

# Develop mode (editable install)
maturin develop

# Or build wheel
maturin build --release
```

## Building with Cargo Directly

If you prefer using cargo directly:

```bash
cd /path/to/statoxide

# Set Python version (adjust as needed)
export PYO3_PYTHON=python3.11

# Build in debug mode
cargo build --package so-python

# Or release mode
cargo build --release --package so-python
```

The shared library will be at:
- Debug: `target/debug/libso_python.so`
- Release: `target/release/libso_python.so`

## Installing the Python Package

### Using pip with wheel:
```bash
pip install target/wheels/statoxide-*.whl
```

### Using develop mode (for development):
```bash
pip install -e .
```

## Troubleshooting

### 1. "unable to find library -lpython3.11"
This means Python development headers are not installed. Install them using your package manager.

### 2. "undefined symbol" errors
Make sure you're using compatible versions of Python and Rust. Check Python version:
```bash
python3 --version
```

### 3. Module import errors
Ensure the `.so` file is in your Python path or install properly with pip/maturin.

## Testing the Python Bindings

After installation:

```python
import statoxide

# Test basic functions
print(statoxide.version())
print(statoxide.stats.mean([1.0, 2.0, 3.0]))

# Test DataFrame
import statoxide.core as soc

df = soc.DataFrame({
    "x": [1.0, 2.0, 3.0],
    "y": [4.0, 5.0, 6.0]
})
print(df)
print(df.columns())

# Test Series
series = df.get_column("x")
print(series.name())
print(series.mean())
```

## Development Notes

- The Python module structure mirrors the Rust crate structure
- Core data structures: `Series`, `DataFrame`, `Formula`
- Statistical functions in `statoxide.stats` module
- Statistical models in `statoxide.models` module  
- Time series analysis in `statoxide.tsa` module
- Utilities in `statoxide.utils` module