#!/bin/bash

set -e  # Exit on error

REQUIRED_PYTHON_VERSION="3.11"
PYTHON_VERSION_FULL="3.11.7"

# Check if Python version is already installed
check_python_version() {
    if command -v python3 &>/dev/null; then
        CURRENT_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        [[ "$CURRENT_VERSION" == "$REQUIRED_PYTHON_VERSION" ]] && echo "Python $REQUIRED_PYTHON_VERSION is already installed." && return 0
    fi
    return 1
}

# Install Python
install_python() {
    case "$(uname -s)" in
        Darwin)
            echo "Installing Python $PYTHON_VERSION_FULL on macOS..."
            [[ ! $(command -v brew) ]] && /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
            brew update && brew install python@3.11 && brew link python@3.11
            ;;
        Linux)
            echo "Installing Python $PYTHON_VERSION_FULL on Linux..."
            sudo apt update
            sudo apt install -y software-properties-common
            sudo add-apt-repository -y ppa:deadsnakes/ppa
            sudo apt update
            sudo apt install -y python3.11 python3.11-venv python3.11-distutils
            ;;
        *)
            echo "Unsupported OS. Exiting."
            exit 1
            ;;
    esac
}

echo "Checking Python version..."
check_python_version || install_python

# Set Python 3.11 as default
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
echo "Python $PYTHON_VERSION_FULL setup complete."
