#!/bin/bash
set -e

echo "Setting up stardew-vision-training ROCm PyTorch ML environment (existing project mode)..."

# Note: No permissions block needed for workspace files.
# By deleting the ubuntu user in the Dockerfile, common-utils creates our user
# with UID/GID that matches the host, giving automatic permission alignment.

WORKSPACE_DIR="/workspaces/stardew-vision-training"

# Fix ownership of AMD's pre-configured venv
echo "Configuring Python virtual environment permissions..."
sudo chown -R $(whoami):$(whoami) /opt/venv

# Generate rocm-provided.txt (list of all packages in the ROCm container venv)
echo "Extracting ROCm-provided packages..."
if [ -f /etc/pip/constraint.txt ]; then
    grep -E "==" /etc/pip/constraint.txt | sort > ${WORKSPACE_DIR}/rocm-provided.txt
else
    uv pip freeze > ${WORKSPACE_DIR}/rocm-provided.txt
fi

# Update system packages
# Note: AMD internal repos (compute-artifactory.amd.com) are unreachable outside
# AMD's network. The errors are expected and won't affect functionality.
echo "Updating system packages (AMD internal repos may show errors - this is expected)..."
sudo apt-get update --allow-releaseinfo-change 2>&1 || true

sudo apt-get install -y --no-upgrade \
    git curl wget build-essential \
    && sudo rm -rf /var/lib/apt/lists/*

# Install development tools
uv pip install --no-cache-dir ruff pre-commit

# ==============================================================================
# --- Virtual environment setup ---
# (Always runs in existing project mode — no .standalone-project check needed)
# ==============================================================================

cd ${WORKSPACE_DIR}

# Detect /opt/venv Python version — must match to avoid binary incompatibility
CONTAINER_PYTHON_VERSION=$(/opt/venv/bin/python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Container Python version: $CONTAINER_PYTHON_VERSION"

if [ ! -d ".venv" ]; then
    echo "Creating project virtual environment with Python $CONTAINER_PYTHON_VERSION..."
    /opt/venv/bin/python -m venv .venv

    # CRITICAL: Verify the venv uses the same Python version
    VENV_PYTHON_VERSION=$(.venv/bin/python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")

    if [ "$VENV_PYTHON_VERSION" != "$CONTAINER_PYTHON_VERSION" ]; then
        echo "ERROR: venv Python version mismatch!"
        echo "  Container /opt/venv: Python $CONTAINER_PYTHON_VERSION"
        echo "  Created .venv: Python $VENV_PYTHON_VERSION"
        echo "  This will cause binary incompatibility with ROCm packages."
        echo "  Please report this at: https://github.com/thesteve0/datascience-template-ROCm/issues"
        exit 1
    fi

    echo "✓ Created .venv with Python $VENV_PYTHON_VERSION"
else
    echo "✓ .venv already exists, skipping creation"
fi

# Create .pth bridge to make ROCm packages accessible from .venv
# This makes torch, numpy, etc. importable without installing them from PyPI.
VENV_SITE_PACKAGES=$(find .venv/lib -type d -name "site-packages" | head -n 1)
CONTAINER_SITE_PACKAGES="/opt/venv/lib/python${CONTAINER_PYTHON_VERSION}/site-packages"

if [ -n "$VENV_SITE_PACKAGES" ]; then
    PTH_FILE="$VENV_SITE_PACKAGES/_rocm_bridge.pth"
    echo "$CONTAINER_SITE_PACKAGES" > "$PTH_FILE"
    echo "✓ Created .pth bridge: $VENV_SITE_PACKAGES/_rocm_bridge.pth -> $CONTAINER_SITE_PACKAGES"
else
    echo "ERROR: Could not find site-packages in .venv"
    exit 1
fi

# Verify ROCm packages are accessible via the bridge
if .venv/bin/python -c "import torch" 2>/dev/null; then
    TORCH_VERSION=$(.venv/bin/python -c "import torch; print(torch.__version__)")
    echo "✓ torch $TORCH_VERSION accessible via .pth bridge"
else
    echo "⚠ Warning: Could not import torch via .pth bridge"
fi

# ==============================================================================
# --- Dependency setup ---
# ==============================================================================

if [ -f "pyproject.toml" ]; then
    echo ""
    echo "Modifying pyproject.toml for ROCm compatibility..."

    # Install TOML tools via pip (avoids triggering a uv sync prematurely)
    .venv/bin/pip install --quiet tomli tomli-w

    # Modify pyproject.toml and determine if --no-install-project is needed.
    # - Adds [tool.uv] exclude-dependencies for all ROCm-provided packages
    #   (skipped if already present, so re-runs are safe)
    # - Detects hatchling build backend without wheel packages configured,
    #   which would cause uv sync to fail when building the editable install
    #
    # Prints "true" to stdout if --no-install-project is needed, "false" otherwise.
    # All informational output goes to stderr so it appears in the terminal.
    NEEDS_NO_INSTALL=$(.venv/bin/python << 'PYEOF'
import sys
from pathlib import Path
import tomli
import tomli_w

PYTHON_VERSION = f"{sys.version_info.major}.{sys.version_info.minor}"
site_packages = Path(f'/opt/venv/lib/python{PYTHON_VERSION}/site-packages')
packages = sorted({
    d.name.split('-')[0].replace('_', '-').lower()
    for d in site_packages.glob('*.dist-info')
})

with open('pyproject.toml', 'rb') as f:
    config = tomli.load(f)

# Add exclude-dependencies only if not already configured (safe to re-run)
if 'tool' not in config:
    config['tool'] = {}
if 'uv' not in config['tool']:
    config['tool']['uv'] = {}
if 'exclude-dependencies' not in config['tool']['uv']:
    config['tool']['uv']['exclude-dependencies'] = packages
    print(f"✓ Added {len(packages)} ROCm packages to exclude-dependencies", file=sys.stderr)
else:
    print("✓ exclude-dependencies already configured, skipping", file=sys.stderr)

# Detect hatchling without wheel packages — this causes uv sync to fail when
# building the editable install. Safe default: use --no-install-project.
build_backend = config.get('build-system', {}).get('build-backend', '')
needs_no_install = False
if 'hatchling' in build_backend:
    wheel_cfg = (
        config.get('tool', {})
              .get('hatch', {})
              .get('build', {})
              .get('targets', {})
              .get('wheel', {})
    )
    if 'packages' not in wheel_cfg:
        needs_no_install = True
        print("⚠ Hatchling detected without [tool.hatch.build.targets.wheel] packages configured",
              file=sys.stderr)

with open('pyproject.toml', 'wb') as f:
    tomli_w.dump(config, f)

# Only "true" or "false" goes to stdout — captured by the shell variable
print("true" if needs_no_install else "false", end="")
PYEOF
    )
    echo "✓ pyproject.toml updated"

    # Run uv sync, with --no-install-project if hatchling wheel config is missing
    echo ""
    echo "Syncing project dependencies..."
    if [ "${NEEDS_NO_INSTALL:-false}" = "true" ]; then
        echo ""
        echo "  ⚠ Running 'uv sync --no-install-project' because hatchling needs"
        echo "    [tool.hatch.build.targets.wheel] to build the editable install."
        echo ""
        echo "    Your dependencies will be installed, but the project itself will"
        echo "    not be installed as an editable package. Scripts and imports that"
        echo "    rely on the package being installed may not resolve correctly."
        echo ""
        echo "    To enable editable install, add to your pyproject.toml:"
        echo "      [tool.hatch.build.targets.wheel]"
        echo "      packages = [\"src/your_package_name\"]"
        echo "    Then run: uv sync"
        echo ""
        uv sync --no-install-project
    else
        uv sync
    fi

elif [ -f "requirements.txt" ]; then
    echo ""
    echo "No pyproject.toml found — using requirements.txt workflow."
    echo "Filtering ROCm-conflicting packages..."
    .venv/bin/python scripts/resolve-dependencies.py requirements.txt
    uv pip install -r requirements-filtered.txt
    echo "✓ Dependencies installed from requirements-filtered.txt"

else
    echo ""
    echo "⚠ No pyproject.toml or requirements.txt found."
    echo "  Add your dependencies and then run one of:"
    echo "    uv sync                       (if you add a pyproject.toml)"
    echo "    uv pip install -r requirements.txt  (if you add a requirements.txt)"
fi

# ==============================================================================
# --- Configure git identity ---
# ==============================================================================

echo ""
echo "Configuring git identity..."
git config --global user.name "Steven Pousty"
git config --global user.email "steve.pousty@gmail.com"
git config --global init.defaultBranch main

# ==============================================================================
# --- Verify ROCm installation ---
# ==============================================================================

echo ""
echo "Verifying ROCm installation..."
if command -v amd-smi &> /dev/null; then
    echo "AMD SMI found. GPU status:"
    amd-smi || echo "Warning: amd-smi failed (this is normal if no GPU is available)"
elif command -v rocm-smi &> /dev/null; then
    echo "ROCm SMI found (note: amd-smi is preferred). GPU status:"
    rocm-smi || echo "Warning: rocm-smi failed (this is normal if no GPU is available)"
else
    echo "Warning: Neither amd-smi nor rocm-smi found in PATH"
fi

echo ""
echo "Setup complete!"
echo "  - .venv created with .pth bridge to ROCm packages"
echo "  - ROCm packages protected in pyproject.toml"
echo "  - Store models in ./models/ and datasets in ./datasets/ — they persist across rebuilds"