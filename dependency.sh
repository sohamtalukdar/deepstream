#!/bin/bash
echo "🚀 Updating package lists..."
apt update
echo "📦 Upgrading pip..."
python3 -m pip install --upgrade pip
pip3 install onnx
pip3 install torch
pip install ultralytics
echo "⚡ Installing cuda-python..."
pip3 install cuda-python
echo "Downloading pyds..."
wget -O pyds-1.2.0-cp310-cp310-linux_x86_64.whl \
  https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/releases/download/v1.2.0/pyds-1.2.0-cp310-cp310-linux_x86_64.whl
echo "📦 Installing pyds..."
pip3 install ./pyds-1.2.0-cp310-cp310-linux_x86_64.whl
pip3 install websockets
pip3 install opencv-python
pip3 install cupy-cuda12x
echo "✅ Installation complete!"
