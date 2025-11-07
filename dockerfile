# ============================================
# ✅ 1. ARM-Compatible Python Base Image
# ============================================
FROM python:3.11-slim

# ============================================
# ✅ 2. Set workdir
# ============================================
WORKDIR /app

# ============================================
# ✅ 3. Install system deps
# ============================================
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    libgl1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ============================================
# ✅ 4. Install TensorFlow for ARM (works on M1/M2)
# ============================================
RUN pip install --upgrade pip && \
    pip install --no-cache-dir tensorflow==2.13.0

# ============================================
# ✅ 5. Copy requirements FIRST
# ============================================
COPY requirements.txt .

# ============================================
# ✅ 6. Install remaining dependencies
# ============================================
RUN pip install --no-cache-dir -r requirements.txt

# ============================================
# ✅ 7. Copy project
# ============================================
COPY . .

# ============================================
# ✅ 8. Copy model
# ============================================
COPY serving/model/e45e81134551430085cf29fe7af76f30/artifacts/lstm_stock_model.keras /app/models/lstm_stock_model.keras
COPY serving/model/e45e81134551430085cf29fe7af76f30/artifacts/standard_scaler.pkl   /app/models/standard_scaler.pkl

# ============================================
# ✅ 9. Environment
# ============================================
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# ============================================
# ✅ 10. Expose ports
# ============================================
EXPOSE 8000  
EXPOSE 8503  
EXPOSE 8502

# ============================================
# ✅ 11. Start all services
# ============================================
CMD ["bash", "-c", "\
    uvicorn api.fastapi_app:app --host 0.0.0.0 --port 8000 & \
    streamlit run app/dashboard_stock.py --server.port 8503 --server.address 0.0.0.0 & \
    streamlit run app/ui_sentiment.py --server.port 8502 --server.address 0.0.0.0 \
"]
