# Tektra Development Guide

## 🚀 Fast Development Workflow

### Quick Commands

```bash
# Ultra-fast development (no bundling) - RECOMMENDED for testing
./dev-run.sh

# Quick build with app bundle (faster than full Tauri build)
./quick-build.sh

# Setup optimized dev environment (run once)
./dev-env-setup.sh

# Traditional Tauri development mode
npm run dev        # Frontend only
cargo tauri dev    # Full Tauri mode
```

## 🎯 Development Modes

### 1. Ultra-Fast Mode (`./dev-run.sh`)
- **Use for**: Testing AI functionality, debugging, quick iterations
- **Speed**: ~10-30 seconds for incremental builds
- **What it does**: Builds Rust backend only, runs directly
- **UI**: Terminal-based interface

### 2. Quick Bundle Mode (`./quick-build.sh`)
- **Use for**: Testing UI integration, creating distributable app
- **Speed**: ~1-2 minutes for full build
- **What it does**: Builds frontend + backend, creates simple .app bundle
- **UI**: Full Tauri UI in app bundle

### 3. Production Mode (`cargo tauri build`)
- **Use for**: Final releases, distribution
- **Speed**: ~5-15 minutes (includes all optimizations)
- **What it does**: Full Tauri build with code signing, DMG creation
- **UI**: Production-ready signed application

## ⚡ Build Speed Optimizations

### Applied Optimizations:
- ✅ Removed slow `beforeBuildCommand` from Tauri config
- ✅ Fixed bundle identifier to avoid warnings
- ✅ Created cargo config for faster compilation
- ✅ Development scripts bypass heavy bundling
- ✅ Incremental builds with caching

### Performance Comparison:
| Mode | First Build | Incremental | Use Case |
|------|-------------|-------------|----------|
| `./dev-run.sh` | 2-3 min | 10-30 sec | Testing, debugging |
| `./quick-build.sh` | 3-5 min | 1-2 min | UI testing, demos |
| `cargo tauri build` | 10-15 min | 5-10 min | Production releases |

## 🔧 Troubleshooting

### Common Issues:
1. **Slow initial build**: Run `./dev-env-setup.sh` once to optimize
2. **Bundle warnings**: Fixed - identifier changed to `com.tektra.desktop`
3. **Frontend build errors**: Run `npm run build` manually first
4. **Ollama issues**: App includes bundled Ollama, no system install needed

### Environment Variables:
```bash
# Optional optimizations
export RUST_LOG=info
export OLLAMA_DEBUG=false
export NODE_ENV=development
```

## 📁 Key Files

- `dev-run.sh` - Ultra-fast development runner
- `quick-build.sh` - Fast bundling script
- `dev-env-setup.sh` - One-time optimization setup
- `.cargo/config.toml` - Rust compilation optimizations
- `src-tauri/tauri.conf.json` - Tauri configuration

## 🎯 Recommended Workflow

1. **First time setup**:
   ```bash
   ./dev-env-setup.sh
   ```

2. **Daily development**:
   ```bash
   ./dev-run.sh  # For AI/backend testing
   ```

3. **UI testing**:
   ```bash
   ./quick-build.sh && open target/Tektra.app
   ```

4. **Production build**:
   ```bash
   npm run build && cargo tauri build
   ```

## ✅ Status

Your Tektra app is now optimized for:
- ✅ Fast development iterations (10-30 seconds)
- ✅ Complete Tauri 2.x compatibility  
- ✅ Bundled Ollama (no user installation required)
- ✅ Enhanced environment setup for reliability
- ✅ Multiple development modes for different needs

The fork/exec issue has been resolved with enhanced environment variables!