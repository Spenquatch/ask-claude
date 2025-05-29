# Release Process

This document outlines the process for releasing new versions of the Ask Claude package.

## Prerequisites

1. **Permissions**: You need maintainer access to the repository
2. **PyPI Account**: Account on [PyPI](https://pypi.org) and [Test PyPI](https://test.pypi.org)
3. **GitHub Environment Setup**: The repository must have two environments configured:
   - `test-pypi` - For Test PyPI releases
   - `pypi` - For production PyPI releases

## Release Types

### 1. Development/Pre-release (Test PyPI)
For testing and validation before official releases.

### 2. Production Release (PyPI)
Official releases for public consumption.

## Step-by-Step Release Process

### 1. Prepare the Release

1. **Ensure all changes are merged to main**
   ```bash
   git checkout main
   git pull origin main
   ```

2. **Run quality checks locally**
   ```bash
   poetry run pre-commit run --all-files
   poetry run pytest
   poetry run mypy ask_claude/
   ```

3. **Update version in pyproject.toml**
   ```bash
   # For patch release (0.1.0 -> 0.1.1)
   poetry version patch

   # For minor release (0.1.1 -> 0.2.0)
   poetry version minor

   # For major release (0.2.0 -> 1.0.0)
   poetry version major

   # For pre-release (0.2.0 -> 0.2.0rc1)
   poetry version prerelease
   ```

4. **Update CHANGELOG.md**
   - Add release date
   - Move unreleased changes to the new version section
   - Follow [Keep a Changelog](https://keepachangelog.com) format

5. **Commit version bump**
   ```bash
   git add pyproject.toml CHANGELOG.md
   git commit -m "chore: bump version to $(poetry version -s)"
   git push origin main
   ```

### 2. Create GitHub Release

1. **Go to [GitHub Releases](https://github.com/Spenquatch/ask-claude/releases)**

2. **Click "Draft a new release"**

3. **Create a new tag**
   - Tag version: `v{version}` (e.g., `v0.1.0`, `v0.2.0rc1`)
   - Target: `main` branch

4. **Fill in release details**
   - **Release title**: `v{version}` (same as tag)
   - **Description**: Copy relevant section from CHANGELOG.md
   - **Pre-release**: Check this box for release candidates

5. **Publish release**
   - This triggers the automated release workflow

### 3. Monitor Release Workflow

1. **Check [GitHub Actions](https://github.com/Spenquatch/ask-claude/actions)**
   - Quality checks must pass
   - Build process creates distribution files
   - Publishing happens automatically based on release type

2. **Release Types**:
   - **Release Candidates** (tags with `rc`): Publish to Test PyPI
   - **Full Releases**: Publish to production PyPI

### 4. Verify Release

1. **Test PyPI Installation** (for pre-releases)
   ```bash
   pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ ask-claude=={version}
   ```

2. **PyPI Installation** (for production releases)
   ```bash
   pip install ask-claude=={version}
   ```

3. **Verify functionality**
   ```bash
   python -c "from ask_claude import __version__; print(__version__)"
   ask-claude --version
   ```

## Manual Release (Emergency)

If automated release fails, you can release manually:

1. **Build the package**
   ```bash
   poetry build
   ```

2. **Upload to Test PyPI**
   ```bash
   poetry publish -r testpypi
   ```

3. **Upload to PyPI**
   ```bash
   poetry publish
   ```

## Rollback Process

If a release has critical issues:

1. **Yank the release on PyPI** (doesn't delete, but prevents new installs)
   - Go to the project page on PyPI
   - Click on the version
   - Click "Yank this release"

2. **Create a patch release** with the fix
   - Follow normal release process
   - Mention the issue in CHANGELOG.md

## Environment Configuration

### GitHub Environments

The repository uses GitHub Environments for deployment protection:

1. **test-pypi**
   - No approval required
   - Used for release candidates

2. **pypi**
   - Manual approval required (optional)
   - Used for production releases

### PyPI Token Configuration

Using Trusted Publishers (recommended):
1. Configure OIDC publishing on PyPI
2. No tokens needed in GitHub Secrets

Alternative (token-based):
1. Generate API tokens on PyPI/Test PyPI
2. Add as repository secrets:
   - `TEST_PYPI_API_TOKEN`
   - `PYPI_API_TOKEN`

## Troubleshooting

### Version Mismatch Error
If the workflow fails with version mismatch:
1. Ensure `poetry version` was run and committed
2. Tag version must match pyproject.toml version

### Build Failures
1. Check that all dependencies are properly specified
2. Ensure no local-only files are referenced

### Publishing Failures
1. Verify PyPI credentials/OIDC setup
2. Check if package name is available
3. Ensure version doesn't already exist

## Best Practices

1. **Always test with Test PyPI first** for significant changes
2. **Use release candidates** for major versions
3. **Keep CHANGELOG.md updated** throughout development
4. **Tag versions consistently** with `v` prefix
5. **Don't skip quality checks** - let automation ensure quality

## Version Numbering

We follow [Semantic Versioning](https://semver.org/):
- **MAJOR.MINOR.PATCH** (e.g., 1.2.3)
- **Pre-releases**: `{version}rc{number}` (e.g., 1.2.0rc1)

### When to increment:
- **PATCH**: Bug fixes, minor improvements (1.2.3 → 1.2.4)
- **MINOR**: New features, backward compatible (1.2.4 → 1.3.0)
- **MAJOR**: Breaking changes (1.3.0 → 2.0.0)
