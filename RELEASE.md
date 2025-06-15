# Release Process

This document describes how to release new versions of Tektra AI Assistant to PyPI.

## Prerequisites

Before creating a release, ensure you have:

1. **PyPI Account**: Register at [pypi.org](https://pypi.org/account/register/)
2. **Test PyPI Account**: Register at [test.pypi.org](https://test.pypi.org/account/register/)
3. **API Tokens**: Create API tokens for both PyPI and Test PyPI
4. **GitHub Secrets**: Add the tokens to GitHub repository secrets

## Setting Up GitHub Secrets

Add these secrets to your GitHub repository (Settings → Secrets and variables → Actions):

### Required Secrets

1. **`PYPI_API_TOKEN`**: Your PyPI API token
   - Go to [PyPI Account Settings](https://pypi.org/manage/account/)
   - Create API token with project scope
   - Copy the token (starts with `pypi-`)

2. **`TEST_PYPI_API_TOKEN`**: Your Test PyPI API token
   - Go to [Test PyPI Account Settings](https://test.pypi.org/manage/account/)
   - Create API token with project scope
   - Copy the token (starts with `pypi-`)

### GitHub Repository Setup

1. Go to your repository on GitHub
2. Settings → Secrets and variables → Actions
3. Click "New repository secret"
4. Add both tokens

## Release Workflow

The release process is fully automated through GitHub Actions. Here's how it works:

### 1. Development and Testing

```bash
# Make your changes
git add .
git commit -m "Add new feature"
git push origin main
```

This triggers the test workflow that:
- Tests on Python 3.9, 3.10, 3.11, 3.12
- Lints and formats code
- Builds frontend
- Runs integration tests

### 2. Create a Release Tag

When ready to release:

```bash
# Create and push a version tag
git tag v1.0.0
git push origin v1.0.0
```

**Tag Format**: Use semantic versioning with `v` prefix:
- `v1.0.0` - Major release
- `v1.0.1` - Patch release
- `v1.1.0` - Minor release
- `v2.0.0-beta.1` - Pre-release

### 3. Automated Release Process

Pushing a tag triggers the full release workflow:

#### Step 1: Testing
- Runs all tests on multiple Python versions
- Validates code quality and imports

#### Step 2: Frontend Build
- Builds the React/Next.js frontend
- Creates optimized production build

#### Step 3: Package Build
- Extracts version from git tag
- Updates `pyproject.toml` with new version
- Builds Python wheel and source distribution
- Validates package with `twine check`

#### Step 4: Test PyPI Publication
- Publishes to Test PyPI first
- Tests installation from Test PyPI
- Validates package works correctly

#### Step 5: GitHub Release
- Creates GitHub release with changelog
- Uploads package files as release assets
- Creates installation scripts for multiple platforms

#### Step 6: PyPI Publication (Manual)
- **Manual trigger required**: Create a GitHub Release to publish to main PyPI
- Publishes to main PyPI
- Tests installation from PyPI

## Manual Release Steps

### 1. Test the Release Locally

```bash
# Build the package locally
python build_package.py

# Test installation
pip install dist/tektra_ai-*.whl
tektra --version
```

### 2. Create GitHub Release

1. Go to GitHub → Releases → "Create a new release"
2. Choose the tag you created
3. Generate release notes or write custom changelog
4. Click "Publish release"

This triggers the PyPI publication workflow.

### 3. Verify Release

After release, verify:

```bash
# Install from PyPI
pip install tektra

# Test functionality
tektra setup
tektra start
```

## Version Management

### Automatic Version Updates

The workflow automatically:
- Extracts version from git tags
- Updates `pyproject.toml`
- Updates `tektra/__init__.py`

### Version Schema

Follow [Semantic Versioning](https://semver.org/):

- **MAJOR**: Breaking changes
- **MINOR**: New features, backwards compatible
- **PATCH**: Bug fixes, backwards compatible

Examples:
- `v1.0.0` → `v1.0.1` (bug fix)
- `v1.0.1` → `v1.1.0` (new feature)
- `v1.1.0` → `v2.0.0` (breaking change)

## Release Checklist

Before creating a release tag:

- [ ] All tests passing
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version compatibility tested
- [ ] Frontend builds successfully
- [ ] Backend tests pass
- [ ] Integration tests pass
- [ ] No security vulnerabilities

## Troubleshooting

### Build Failures

If the workflow fails:

1. Check the GitHub Actions logs
2. Fix the issue
3. Delete the tag: `git tag -d v1.0.0 && git push --delete origin v1.0.0`
4. Create a new tag with the same version after fixes

### PyPI Upload Failures

Common issues:
- **Version already exists**: Increment version number
- **Package validation failed**: Check `twine check dist/*` output
- **Authentication failed**: Verify API tokens in GitHub secrets

### Frontend Build Issues

If frontend build fails:
- Check Node.js version compatibility
- Verify `package.json` and dependencies
- Test locally: `cd frontend && npm ci && npm run build`

## Rollback Process

If a release has issues:

### 1. Emergency Hotfix

```bash
# Create hotfix branch
git checkout -b hotfix/v1.0.1
# Fix the issue
git commit -m "Fix critical issue"
git tag v1.0.1
git push origin v1.0.1
```

### 2. PyPI Rollback

PyPI doesn't support deleting releases, but you can:
- Yank the problematic version: Use PyPI web interface
- Release a new fixed version immediately

## Advanced Configuration

### Custom Release Notes

Edit `.github/workflows/publish.yml` to customize:
- Release notes format
- Asset naming
- Publication conditions

### Environment-Specific Builds

For different environments:
- Development: Push to `develop` branch
- Staging: Create `staging/*` tags  
- Production: Create `v*` tags

## Support

For release process issues:
- Check GitHub Actions logs
- Review PyPI project page
- Contact maintainers via GitHub issues

---

**Remember**: Always test releases thoroughly before publishing to main PyPI!