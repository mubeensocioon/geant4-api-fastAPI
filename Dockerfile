FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    libxerces-c-dev \
    qtbase5-dev \
    qtchooser \
    qt5-qmake \
    qtbase5-dev-tools \
    libqt5opengl5-dev \
    libexpat1-dev \
    zlib1g-dev \
    xvfb \
    x11-utils \
    imagemagick \
    ghostscript \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    libxmu-dev \
    libxi-dev \
    && rm -rf /var/lib/apt/lists/*

# Fix ImageMagick security policy
RUN sed -i 's/<policy domain="coder" rights="none" pattern="PS" \/>/<policy domain="coder" rights="read|write" pattern="PS" \/>/' /etc/ImageMagick-6/policy.xml && \
    sed -i 's/<policy domain="coder" rights="none" pattern="EPS" \/>/<policy domain="coder" rights="read|write" pattern="EPS" \/>/' /etc/ImageMagick-6/policy.xml && \
    sed -i 's/<policy domain="coder" rights="none" pattern="PDF" \/>/<policy domain="coder" rights="read|write" pattern="PDF" \/>/' /etc/ImageMagick-6/policy.xml

# Create working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Set environment variables for Geant4
ENV G4INSTALL=/root/geant4-v11.3.2-install
ENV PATH=$G4INSTALL/bin:$PATH
ENV LD_LIBRARY_PATH=$G4INSTALL/lib:$LD_LIBRARY_PATH

# Set Qt and display environment variables
ENV QT_QPA_PLATFORM=offscreen
ENV DISPLAY=:99

# Copy application code
COPY geant4_api.py .

# Create directories
RUN mkdir -p /app/outputs

# Create comprehensive startup script
RUN cat > /app/start.sh << 'EOF' && \
#!/bin/bash

echo "🚀 Starting Geant4 Nuclear Simulation API"
echo "=========================================="

# Check Geant4 installation
if [ ! -d "$G4INSTALL" ]; then
    echo "❌ ERROR: Geant4 installation directory not found: $G4INSTALL"
    echo "Please ensure Geant4 is properly mounted or installed"
    echo "Expected location: $G4INSTALL"
    echo "Make sure to mount your host Geant4 installation to this path"
    exit 1
fi

echo "✅ Geant4 installation found: $G4INSTALL"

# List available executables
echo "📋 Available Geant4 executables:"
if [ -d "$G4INSTALL/bin" ]; then
    ls -la "$G4INSTALL/bin/" | grep -E "(geant4|G4)" || echo "   No obvious Geant4 executables found"
else
    echo "   Bin directory not found"
fi

# Verify Qt support in Geant4
echo "🔧 Checking Geant4 Qt support..."
if [ -f "$G4INSTALL/lib/libG4OpenGL.so" ] || [ -f "$G4INSTALL/lib64/libG4OpenGL.so" ]; then
    echo "✅ Geant4 OpenGL support found"
else
    echo "⚠️  Geant4 OpenGL support might be missing"
fi

if [ -f "$G4INSTALL/lib/libG4vis_management.so" ] || [ -f "$G4INSTALL/lib64/libG4vis_management.so" ]; then
    echo "✅ Geant4 visualization management found"
else
    echo "⚠️  Geant4 visualization management might be missing"
fi

# Start virtual display
echo "🖥️  Starting virtual display..."
Xvfb :99 -screen 0 1024x768x24 -ac +extension GLX +render -noreset &
XVFB_PID=$!
export DISPLAY=:99
sleep 3

# Test display
if xdpyinfo -display :99 >/dev/null 2>&1; then
    echo "✅ Virtual display :99 is running"
else
    echo "⚠️  Virtual display might not be working properly"
fi

# Test Qt
echo "🔧 Testing Qt availability..."
qmake --version && echo "✅ Qt is available" || echo "❌ Qt not found"

# Set up cleanup function
cleanup() {
    echo "🧹 Cleaning up..."
    if [ ! -z "$XVFB_PID" ]; then
        kill $XVFB_PID 2>/dev/null
    fi
    exit
}

trap cleanup SIGTERM SIGINT

echo "🌐 Starting FastAPI server..."
exec uvicorn geant4_api:app --host 0.0.0.0 --port 8000 --workers 1
EOF

RUN chmod +x /app/start.sh

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["/app/start.sh"]
