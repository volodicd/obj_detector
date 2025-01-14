#!/bin/bash

# Exit on error
set -e

REQUIRED_PYTHON_VERSION="3.11"
PYTHON_VERSION_FULL="3.11.7"  # Latest stable 3.11 version

# Function to check if command exists
command_exists() {
    command -v "$1" &> /dev/null
}

# Function to check Python version
check_python_version() {
    if command_exists python3; then
        CURRENT_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        if [ "$CURRENT_VERSION" = "$REQUIRED_PYTHON_VERSION" ]; then
            echo "Python $REQUIRED_PYTHON_VERSION is already installed"
            return 0
        fi
    fi
    return 1
}

# Function to detect OS
detect_os() {
    if [ "$(uname)" == "Darwin" ]; then
        echo "macos"
    elif [ -f /etc/os-release ]; then
        source /etc/os-release
        echo "$ID"
    else
        echo "unknown"
    fi
}

# Install Python based on OS
install_python() {
    OS=$(detect_os)

    case $OS in
        "ubuntu"|"debian")
            echo "Installing on Ubuntu/Debian..."
            sudo apt update
            sudo apt install -y wget build-essential libssl-dev zlib1g-dev \
                libbz2-dev libreadline-dev libsqlite3-dev curl \
                libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev \
                libffi-dev liblzma-dev

            # Download and install Python
            wget https://www.python.org/ftp/python/${PYTHON_VERSION_FULL}/Python-${PYTHON_VERSION_FULL}.tgz
            tar xzf Python-${PYTHON_VERSION_FULL}.tgz
            cd Python-${PYTHON_VERSION_FULL}
            ./configure --enable-optimizations
            make -j $(nproc)
            sudo make altinstall
            cd ..
            rm -rf Python-${PYTHON_VERSION_FULL}*
            ;;

        "macos")
            echo "Installing on macOS..."
            if ! command_exists brew; then
                echo "Installing Homebrew first..."
                /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
            fi
            brew update
            brew install python@3.11
            brew link python@3.11
            ;;

        *)
            echo "Unsupported operating system"
            exit 1
            ;;
    esac
}

# Main script
echo "Checking Python version..."

if check_python_version; then
    echo "Correct Python version is already installed"
else
    echo "Python $REQUIRED_PYTHON_VERSION is not installed"
    echo "Installing Python $REQUIRED_PYTHON_VERSION..."
    install_python
fi

# Verify installation
if check_python_version; then
    echo "Python $REQUIRED_PYTHON_VERSION is successfully installed!"
    python3 --version
else
    echo "Failed to install Python $REQUIRED_PYTHON_VERSION"
    exit 1
fi

# Make Python 3.11 the default python3 (might require password)
sudo update-alternatives --install /usr/bin/python3 python3 $(which python${REQUIRED_PYTHON_VERSION}) 1