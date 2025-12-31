ARG BUILD_FROM
FROM ${BUILD_FROM}

# Install system dependencies (Alpine packages)
RUN apk add --no-cache \
    python3 \
    py3-pip \
    python3-dev \
    build-base \
    cmake \
    mesa-gl \
    glib \
    libsm \
    libxext \
    libxrender \
    libgomp \
    jpeg-dev \
    zlib-dev \
    freetype-dev

# Install Python packages
RUN pip3 install --no-cache-dir --break-system-packages \
    fastapi \
    uvicorn[standard] \
    deepface \
    tf-keras \
    python-multipart \
    pillow \
    numpy

# Copy application
COPY run.sh /
COPY server.py /

RUN chmod a+x /run.sh

CMD [ "/run.sh" ]
