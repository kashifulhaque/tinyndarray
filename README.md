# **[WIP] A tiny numpy-like library for python, written in rust**
> To learn the inner workings of numpy

## **Prerequisites**

### **1. Install OpenBLAS**
This project depends on OpenBLAS for linear algebra operations.

**Ubuntu/Debian:**
```bash
sudo apt install libopenblas-dev
```

**Fedora/RHEL:**
```bash
sudo dnf install openblas-devel
```

**Arch Linux:**
```bash
sudo pacman -S openblas
```

**macOS (Homebrew):**
```bash
brew install openblas
```

**macOS/Linux (Nix):**
```bash
nix-shell -p openblas
```

### **2. Install Rust**
Make sure you have the Rust toolchain installed: https://www.rust-lang.org/tools/install

### **3. Install maturin**
Maturin is required to build this Rust-Python extension:

```bash
# Using uv (recommended)
uv tool install maturin

# Or using pip
pip install maturin
```

## **Setup**

1. Clone this repo
2. Create a virtual environment (optional but recommended)
3. Build and install the package:

```bash
# Using maturin directly
maturin develop --release

# Or using uv
uv run maturin develop --release
```

## **Using mise (optional)**

If you have [mise](https://mise.jdx.dev/) installed, you can use the provided tasks:

```bash
# Build and install in development mode
mise run develop

# Run tests
mise run test

# Run benchmarks
mise run bench

# Check the code
mise run check
```

## **Manual Execution**

After building, you can run:
- [`bench_matmul.py`](./bench_matmul.py) - Matrix multiplication benchmarks
- [`test.py`](./test.py) - Basic functionality tests
