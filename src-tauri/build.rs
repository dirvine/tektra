use std::path::PathBuf;

fn main() {
    // Standard Tauri build
    tauri_build::build();
    
    // Tell Cargo to rerun if dist folder changes
    println!("cargo:rerun-if-changed=dist");
    
    // Ensure dist folder exists
    let dist_path = PathBuf::from("../dist");
    if !dist_path.exists() {
        panic!("dist folder not found. Please run 'npm run build' first.");
    }
}