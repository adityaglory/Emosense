# 1. Base Image: Python 3.10 versi Slim (Ringan)
FROM python:3.10-slim

# 2. Set Working Directory
WORKDIR /app

# 3. Copy Requirements dulu (biar cache layer optimal)
COPY requirements.txt .

# 4. Install Dependencies
# Kita gabungin command biar layernya dikit
# Install build-essential jaga-jaga kalau ada library butuh compile C++
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir -r requirements.txt

# 5. Copy Seluruh Source Code & Model
COPY . .

# 6. Expose Port Streamlit (Default 8501)
EXPOSE 8501

# 7. Command Utama untuk Menjalankan Aplikasi
CMD ["streamlit", "run", "src/app.py", "--server.address=0.0.0.0"]