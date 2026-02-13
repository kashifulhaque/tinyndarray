fn main() {
    // Allow an explicit override of the OpenBLAS library directory.
    if let Ok(custom_dir) = std::env::var("OPENBLAS_LIB_DIR") {
        if std::path::Path::new(&custom_dir).exists() {
            println!("cargo:rustc-link-search={custom_dir}");
            println!("cargo:rustc-link-lib=openblas");
            return;
        }
    }

    // Common OpenBLAS paths by platform
    let search_paths = [
        "/opt/homebrew/opt/openblas/lib",    // macOS (Apple Silicon, Homebrew)
        "/usr/local/opt/openblas/lib",       // macOS (Intel, Homebrew)
        "/usr/lib/x86_64-linux-gnu",         // Ubuntu/Debian x86_64
        "/usr/lib/aarch64-linux-gnu",        // Ubuntu/Debian aarch64
        "/usr/lib64",                        // Fedora/RHEL
        "/usr/lib",                          // Arch Linux / generic
    ];

    for path in &search_paths {
        if std::path::Path::new(path).exists() {
            println!("cargo:rustc-link-search={path}");
            println!("cargo:rustc-link-lib=openblas");
            return;
        }
    }

    // Fallback: rely on system default library search paths.
    println!("cargo:rustc-link-lib=openblas");
}
