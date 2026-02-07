fn main() {
    println!("cargo:rustc-link-search=/opt/homebrew/opt/openblas/lib");
    println!("cargo:rustc-link-lib=openblas");
}
