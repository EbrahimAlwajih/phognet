# GitHub Releases (no PyPI)

This project is packaged as a Python library **inside the repository**, but it is **not published to PyPI**.
Users install it from source (recommended) or from a release asset.

## Option A (recommended): install from source
```bash
git clone https://github.com/<org-or-user>/phog-net.git
cd phog-net
pip install -e .
```

## Option B: install from a GitHub Release asset (zip)
1. Download the `phog-net-<version>.zip` from the GitHub Release page.
2. Unzip it and install:
```bash
unzip phog-net-<version>.zip
cd phog-net-<version>
pip install .
```

## Creating a GitHub Release (manual)
1. Update version:
   - `pyproject.toml` → `project.version`
   - `src/phognet/__init__.py` → `__version__`
2. Commit and tag:
```bash
git commit -am "Release v0.1.1"
git tag v0.1.1
git push origin main --tags
```
3. On GitHub:
   - Go to **Releases** → **Draft a new release**
   - Choose tag `v0.1.1`
   - Title: `v0.1.1`
   - Attach artifacts (optional): `phog-net-0.1.1.zip` (see workflow below)

## Automated release assets (recommended)
This repo includes a GitHub Actions workflow that:
- runs CI on tag pushes
- builds a source distribution and wheel (optional but useful)
- creates a `.zip` source snapshot as a release asset
- publishes a GitHub Release using the tag name

Workflow file: `.github/workflows/release.yml`

### Trigger
Push a tag like `v0.1.1`.

### Artifacts
- `dist/*.whl` and `dist/*.tar.gz`
- `phog-net-<tag>.zip`

> Note: This still does **not** upload to PyPI.
