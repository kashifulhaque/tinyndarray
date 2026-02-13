fn main() {
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
        }
    }

    println!("cargo:rustc-link-lib=openblas");
}
