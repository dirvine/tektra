use std::process::Command;
use std::env;

pub fn run_cli() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    
    if args.len() > 1 {
        match args[1].as_str() {
            "help" | "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            "version" | "--version" | "-v" => {
                println!("Tektra v{}", env!("CARGO_PKG_VERSION"));
                return Ok(());
            }
            "dev" => {
                println!("Starting Tektra in development mode...");
                run_dev_mode()?;
                return Ok(());
            }
            _ => {
                eprintln!("Unknown command: {}", args[1]);
                print_help();
                return Ok(());
            }
        }
    }
    
    // Default: run the app normally
    Ok(())
}

fn print_help() {
    println!(r#"
Tektra - Voice-Interactive AI Assistant

USAGE:
    tektra [COMMAND]

COMMANDS:
    help, -h, --help      Show this help message
    version, -v, --version Show version information
    dev                   Run in development mode with hot reload

EXAMPLES:
    tektra              # Run the application
    tektra dev          # Run in development mode
    tektra --version    # Show version

For more information, visit: https://github.com/dirvine/tektra
"#);
}

fn run_dev_mode() -> Result<(), Box<dyn std::error::Error>> {
    // Run npm tauri dev
    let mut cmd = Command::new("npm")
        .args(&["run", "tauri", "dev"])
        .spawn()?;
    
    cmd.wait()?;
    Ok(())
}