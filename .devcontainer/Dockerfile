# Use official Python 3.10 image with Node.js for Prisma
FROM python:3.10-bullseye

# Install Node.js (required for Prisma)
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy requirements first for better caching
COPY requirements.txt .
COPY environment.yml .

# Install Python dependencies
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Install Prisma CLI globally
RUN npm install -g prisma

# Create non-root user
ARG USERNAME=msrk
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && chown -R $USERNAME:$USERNAME /workspace

USER $USERNAME

# Set default shell
SHELL ["/bin/bash", "-c"]
