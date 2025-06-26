# Gunakan image Python resmi
FROM python:3.11-slim

# Install dependencies sistem (OpenCV, libGL, dsb)
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy semua file ke container
COPY . .

# Install dependencies Python
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose port (Railway pakai $PORT)
EXPOSE 8080

# Jalankan aplikasi dengan gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]
