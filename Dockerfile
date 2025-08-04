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
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    libxt-dev \
    libxmu-dev \
    libxi-dev \
    qtbase5-dev \
    libexpat1-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    imagemagick \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    libxmu-dev \
    libxi-dev \
    imagemagick \
    && rm -rf /var/lib/apt/lists/*


# Install ghostscript (required for PS/EPS conversion)
RUN apt-get update && apt-get install -y \
    ghostscript \
    && rm -rf /var/lib/apt/lists/*

# Fix ImageMagick security policy for EPS/PS conversion
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

# Create directories
RUN mkdir -p /app/outputs

# Set environment variables for Geant4 (will be mounted as volume)
ENV G4INSTALL=/root/geant4-v11.3.2-install
ENV PATH=$G4INSTALL/bin:$PATH
ENV LD_LIBRARY_PATH=$G4INSTALL/lib:$LD_LIBRARY_PATH

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "geant4_api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
