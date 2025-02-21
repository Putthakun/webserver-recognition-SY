FROM python:3.9-slim

# ตั้งค่า working directory
WORKDIR /app

# ติดตั้ง dependencies ที่จำเป็น
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    unixodbc-dev \
    curl \
    apt-transport-https \
    ca-certificates \
    gnupg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ติดตั้ง Microsoft ODBC Driver 17 สำหรับ SQL Server
RUN curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - \
    && curl https://packages.microsoft.com/config/debian/10/prod.list > /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install -y msodbcsql17 \
    && rm -rf /var/lib/apt/lists/*

COPY wait-for-it.sh /app/wait-for-it.sh
RUN chmod +x /app/wait-for-it.sh
RUN ls -l /app  # ตรวจสอบว่าไฟล์ถูก copy แล้วจริงๆ
# คัดลอก requirements.txt และติดตั้ง dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# คัดลอกไฟล์แอปพลิเคชัน
COPY ./app ./app

# ตั้งค่า PYTHONPATH (ถ้าต้องการ)
ENV PYTHONPATH=/app

RUN ls -R /app

# เปิด port สำหรับ FastAPI
EXPOSE 8001

# คำสั่งรันแอปพลิเคชัน
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
