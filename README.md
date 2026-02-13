# **[WIP] ferray — A tiny NumPy-like library for Python, written in Rust**

> To learn the inner workings of NumPy

## Project Structure

```
ferray/
├── src/                  # Rust source code
│   ├── lib.rs            # Module entry point & PyO3 module definition
│   ├── array.rs          # NdArray class (core data structure)
│   ├── operations.rs     # Element-wise & matrix operations (add, mul, matmul, etc.)
│   ├── conversions.rs    # Conversions between Python lists, NumPy arrays & NdArray
│   └── utils.rs          # Stride computation, broadcasting, index utilities
├── tests/                # Python tests
│   ├── test_basic.py     # Core functionality tests
│   └── test_shapes.py    # Matrix multiplication shape tests
├── benchmarks/           # Performance benchmarks
│   └── bench_matmul.py   # Matrix multiplication benchmark (ferray vs NumPy)
├── Cargo.toml            # Rust package manifest
├── pyproject.toml        # Python package metadata (maturin build system)
├── build.rs              # Build script (OpenBLAS linking)
├── Dockerfile            # Container image for building & running
└── docker-compose.yml    # Docker Compose services (dev, test, bench)
```

## Prerequisites

### 1. Install OpenBLAS

This project depends on OpenBLAS for linear algebra operations.

**Ubuntu/Debian:**
```bash
sudo apt-get install libopenblas-dev
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

### 2. Install Rust

Make sure you have the Rust toolchain installed: https://www.rust-lang.org/tools/install

### 3. Install maturin

Maturin is required to build this Rust-Python extension:

```bash
# Using uv (recommended)
uv tool install maturin

# Or using pip
pip install maturin
```

## Building Locally

```bash
# Clone and enter the repository
git clone https://github.com/kashifulhaque/ferray.git
cd ferray

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# Install dependencies
pip install maturin numpy

# Build and install in development mode
maturin develop --release
```

### Using uv (alternative)

```bash
uv run maturin develop --release --uv
```

## Running Locally

After building, you can run:

```bash
# Run basic tests
python tests/test_basic.py

# Run shape-specific matmul tests
python tests/test_shapes.py

# Run benchmarks
python benchmarks/bench_matmul.py
```

## Using Docker

Build and run everything in a container — no local Rust or OpenBLAS needed:

```bash
# Run tests
docker compose run test

# Run benchmarks
docker compose run bench

# Interactive development
docker compose run dev
```

## Using mise (optional)

If you have [mise](https://mise.jdx.dev/) installed, you can use the provided tasks:

```bash
mise run check     # cargo check
mise run build     # cargo build
mise run develop   # Build & install with maturin
mise run test      # Run all Python tests
mise run bench     # Run benchmarks
```

## CI/CD

This project uses GitHub Actions for:

- **CI** (`ci.yml`): Runs `cargo check`, `clippy`, and `fmt` on every push/PR, then builds wheels for Linux and macOS.
- **Benchmarks** (`benchmarks.yml`): Runs matmul benchmarks on push to `main` and publishes results to the repository wiki.
