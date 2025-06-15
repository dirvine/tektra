#!/usr/bin/env python3
"""
Tektra Development Environment Manager

This script provides easy commands for managing the Tektra development environment.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
DOCKER_DIR = PROJECT_ROOT / "docker"
BACKEND_DIR = PROJECT_ROOT / "backend"
FRONTEND_DIR = PROJECT_ROOT / "frontend"


def run_command(cmd: str, cwd: Path = None, check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command and return the result."""
    print(f"🔧 Running: {cmd}")
    if cwd:
        print(f"📁 In directory: {cwd}")
    
    return subprocess.run(
        cmd, 
        shell=True, 
        cwd=cwd, 
        check=check,
        capture_output=False
    )


def docker_compose(action: str, *args) -> subprocess.CompletedProcess:
    """Run docker-compose command."""
    cmd = f"docker-compose {action} {' '.join(args)}"
    return run_command(cmd, cwd=DOCKER_DIR)


def start_services(services: list = None):
    """Start all or specific services."""
    print("🚀 Starting Tektra development environment...")
    
    if services:
        docker_compose("up", "-d", *services)
        print(f"✅ Started services: {', '.join(services)}")
    else:
        docker_compose("up", "-d")
        print("✅ Started all services")
    
    print("\n🌐 Access points:")
    print("  • Frontend:  http://localhost:3000")
    print("  • Backend:   http://localhost:8000")
    print("  • API Docs:  http://localhost:8000/docs")
    print("  • Database:  localhost:5432")
    print("  • Redis:     localhost:6379")


def stop_services():
    """Stop all services."""
    print("🛑 Stopping Tektra development environment...")
    docker_compose("down")
    print("✅ Stopped all services")


def restart_services(services: list = None):
    """Restart all or specific services."""
    print("🔄 Restarting services...")
    
    if services:
        docker_compose("restart", *services)
        print(f"✅ Restarted services: {', '.join(services)}")
    else:
        docker_compose("restart")
        print("✅ Restarted all services")


def rebuild_services(services: list = None):
    """Rebuild and restart services."""
    print("🔨 Rebuilding services...")
    
    if services:
        docker_compose("up", "-d", "--build", *services)
        print(f"✅ Rebuilt services: {', '.join(services)}")
    else:
        docker_compose("up", "-d", "--build")
        print("✅ Rebuilt all services")


def show_logs(service: str = None, follow: bool = False):
    """Show logs for all or specific service."""
    if follow:
        if service:
            docker_compose("logs", "-f", service)
        else:
            docker_compose("logs", "-f")
    else:
        if service:
            docker_compose("logs", "--tail=50", service)
        else:
            docker_compose("logs", "--tail=50")


def show_status():
    """Show status of all services."""
    print("📊 Service status:")
    docker_compose("ps")
    
    print("\n💾 Volume usage:")
    run_command("docker volume ls | grep docker_")
    
    print("\n📈 Resource usage:")
    run_command("docker stats --no-stream")


def clean_environment(full: bool = False):
    """Clean Docker environment."""
    print("🧹 Cleaning Docker environment...")
    
    # Stop all services
    docker_compose("down")
    
    if full:
        print("🗑️  Removing volumes and data...")
        docker_compose("down", "-v")
        run_command("docker system prune -f")
        print("✅ Full cleanup completed")
    else:
        run_command("docker system prune -f")
        print("✅ Basic cleanup completed")


def setup_env():
    """Set up environment files."""
    env_example = PROJECT_ROOT / ".env.example"
    env_file = PROJECT_ROOT / ".env"
    
    if not env_file.exists() and env_example.exists():
        print("📝 Creating .env file from template...")
        import shutil
        shutil.copy(env_example, env_file)
        print("✅ Created .env file")
        print("⚠️  Please review and update the .env file with your settings")
    else:
        print("✅ Environment file already exists")


def run_backend_command(command: str):
    """Run a command in the backend container."""
    print(f"🐍 Running backend command: {command}")
    run_command(f"docker-compose exec backend {command}", cwd=DOCKER_DIR)


def run_frontend_command(command: str):
    """Run a command in the frontend container."""
    print(f"⚛️  Running frontend command: {command}")
    run_command(f"docker-compose exec frontend {command}", cwd=DOCKER_DIR)


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Tektra Development Environment Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python dev.py start                    # Start all services
  python dev.py start backend frontend  # Start specific services
  python dev.py stop                     # Stop all services
  python dev.py restart backend         # Restart backend service
  python dev.py rebuild                 # Rebuild all services
  python dev.py logs backend -f         # Follow backend logs
  python dev.py status                  # Show service status
  python dev.py clean                   # Basic cleanup
  python dev.py clean --full           # Full cleanup with data removal
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Start command
    start_parser = subparsers.add_parser("start", help="Start services")
    start_parser.add_argument("services", nargs="*", help="Specific services to start")
    
    # Stop command
    subparsers.add_parser("stop", help="Stop all services")
    
    # Restart command
    restart_parser = subparsers.add_parser("restart", help="Restart services")
    restart_parser.add_argument("services", nargs="*", help="Specific services to restart")
    
    # Rebuild command
    rebuild_parser = subparsers.add_parser("rebuild", help="Rebuild and restart services")
    rebuild_parser.add_argument("services", nargs="*", help="Specific services to rebuild")
    
    # Logs command
    logs_parser = subparsers.add_parser("logs", help="Show service logs")
    logs_parser.add_argument("service", nargs="?", help="Specific service")
    logs_parser.add_argument("-f", "--follow", action="store_true", help="Follow logs")
    
    # Status command
    subparsers.add_parser("status", help="Show service status")
    
    # Clean command
    clean_parser = subparsers.add_parser("clean", help="Clean Docker environment")
    clean_parser.add_argument("--full", action="store_true", help="Remove volumes and data")
    
    # Setup command
    subparsers.add_parser("setup", help="Set up environment files")
    
    # Backend command
    backend_parser = subparsers.add_parser("backend", help="Run backend command")
    backend_parser.add_argument("cmd", help="Command to run")
    
    # Frontend command
    frontend_parser = subparsers.add_parser("frontend", help="Run frontend command")
    frontend_parser.add_argument("cmd", help="Command to run")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == "start":
            setup_env()
            start_services(args.services)
        elif args.command == "stop":
            stop_services()
        elif args.command == "restart":
            restart_services(args.services)
        elif args.command == "rebuild":
            rebuild_services(args.services)
        elif args.command == "logs":
            show_logs(args.service, args.follow)
        elif args.command == "status":
            show_status()
        elif args.command == "clean":
            clean_environment(args.full)
        elif args.command == "setup":
            setup_env()
        elif args.command == "backend":
            run_backend_command(args.cmd)
        elif args.command == "frontend":
            run_frontend_command(args.cmd)
        else:
            parser.print_help()
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\n⏹️  Operation cancelled")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()