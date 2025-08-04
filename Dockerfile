FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies for Geant4 visualization
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
    && rm -rf /var/lib/apt/lists/*

# Install OpenGL and X11 dependencies for OGLSX visualization
RUN apt-get update && apt-get install -y \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    libxt-dev \
    libxmu-dev \
    libxi-dev \
    libx11-dev \
    libxext-dev \
    libxrender-dev \
    libxrandr-dev \
    libxss-dev \
    libxinerama-dev \
    libxcursor-dev \
    libxcomposite-dev \
    libxdamage-dev \
    libxfixes-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Qt5 for Geant4 GUI (optional but recommended)
RUN apt-get update && apt-get install -y \
    qtbase5-dev \
    qtchooser \
    qt5-qmake \
    qtbase5-dev-tools \
    libqt5opengl5-dev \
    && rm -rf /var/lib/apt/lists/*

# Install additional libraries needed by Geant4
RUN apt-get update && apt-get install -y \
    libexpat1-dev \
    zlib1g-dev \
    libxerces-c-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Xvfb for headless X11 server (essential for OGLSX)
RUN apt-get update && apt-get install -y \
    xvfb \
    x11-utils \
    x11-xserver-utils \
    && rm -rf /var/lib/apt/lists/*

# Install ImageMagick for image conversion
RUN apt-get update && apt-get install -y \
    imagemagick \
    && rm -rf /var/lib/apt/lists/*

# Install Ghostscript (required for EPS/PS conversion)
RUN apt-get update && apt-get install -y \
    ghostscript \
    && rm -rf /var/lib/apt/lists/*

# Install additional tools
RUN apt-get update && apt-get install -y \
    vim \
    htop \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Fix ImageMagick security policy for EPS/PS/PDF conversion
RUN sed -i 's/<policy domain="coder" rights="none" pattern="PS" \/>/<policy domain="coder" rights="read|write" pattern="PS" \/>/' /etc/ImageMagick-6/policy.xml && \
    sed -i 's/<policy domain="coder" rights="none" pattern="EPS" \/>/<policy domain="coder" rights="read|write" pattern="EPS" \/>/' /etc/ImageMagick-6/policy.xml && \
    sed -i 's/<policy domain="coder" rights="none" pattern="PDF" \/>/<policy domain="coder" rights="read|write" pattern="PDF" \/>/' /etc/ImageMagick-6/policy.xml

# Create working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY geant4_api.py .

# Create necessary directories
RUN mkdir -p /app/outputs/images /app/outputs/temp

# Set proper permissions for output directories
RUN chmod 755 /app/outputs && \
    chmod 755 /app/outputs/images && \
    chmod 755 /app/outputs/temp

# Set environment variables for Geant4
ENV G4INSTALL=/root/geant4-v11.3.2-install
ENV PATH=$G4INSTALL/bin:$PATH
ENV LD_LIBRARY_PATH=$G4INSTALL/lib:$LD_LIBRARY_PATH

# Set environment variables for X11 and OpenGL
ENV DISPLAY=:99
ENV MESA_GL_VERSION_OVERRIDE=3.3
ENV LIBGL_ALWAYS_SOFTWARE=1

# Create startup script for Xvfb
RUN echo '#!/bin/bash\n\
# Start Xvfb in background\n\
Xvfb :99 -screen 0 1024x768x24 -ac +extension GLX +render -noreset &\n\
XVFB_PID=$!\n\
\n\
# Wait for Xvfb to start\n\
sleep 2\n\
\n\
# Start the FastAPI application\n\
uvicorn geant4_api:app --host 0.0.0.0 --port 8000 --workers 1\n\
\n\
# Cleanup on exit\n\
trap "kill $XVFB_PID" EXIT\n' > /app/start.sh && \
    chmod +x /app/start.sh

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Use the startup script
CMD ["/app/start.sh"]
