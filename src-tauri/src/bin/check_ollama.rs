use std::process::Command;

fn main() {
    println!("🔍 Checking Ollama status...\n");
    
    // Check if ollama command exists
    println!("1. Checking for system Ollama command:");
    match Command::new("which").arg("ollama").output() {
        Ok(output) => {
            if output.status.success() {
                let path = String::from_utf8_lossy(&output.stdout);
                println!("   ✅ Found at: {}", path.trim());
                
                // Check version
                if let Ok(version_output) = Command::new("ollama").arg("--version").output() {
                    let version = String::from_utf8_lossy(&version_output.stdout);
                    println!("   Version: {}", version.trim());
                }
            } else {
                println!("   ❌ Ollama command not found in PATH");
            }
        }
        Err(e) => println!("   ❌ Error checking: {}", e),
    }
    
    // Check if Ollama server is running
    println!("\n2. Checking if Ollama server is running:");
    match std::process::Command::new("curl")
        .args(&["-s", "http://localhost:11434/api/tags"])
        .output()
    {
        Ok(output) => {
            if output.status.success() {
                println!("   ✅ Ollama server is running on port 11434");
                let response = String::from_utf8_lossy(&output.stdout);
                if response.contains("models") {
                    println!("   Server response looks valid");
                }
            } else {
                println!("   ❌ Ollama server is not responding on port 11434");
            }
        }
        Err(e) => println!("   ❌ Error checking server: {}", e),
    }
    
    // Check temp directory permissions
    println!("\n3. Checking temp directory:");
    let temp_dir = std::env::temp_dir();
    println!("   Temp dir: {:?}", temp_dir);
    
    let tektra_dir = temp_dir.join("tektra_ollama");
    match std::fs::create_dir_all(&tektra_dir) {
        Ok(_) => {
            println!("   ✅ Can create tektra_ollama directory");
            // Clean up
            let _ = std::fs::remove_dir(&tektra_dir);
        }
        Err(e) => println!("   ❌ Cannot create directory: {}", e),
    }
    
    println!("\n4. Environment info:");
    println!("   OS: {}", std::env::consts::OS);
    println!("   ARCH: {}", std::env::consts::ARCH);
    
    println!("\nIf Ollama is not installed, Tektra will download and use a bundled version.");
}