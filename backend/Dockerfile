FROM python:3.11-slim

# 设为 true 可在镜像内安装 MinerU (体积较大, 也可选择在宿主机/独立容器运行 MinerU)
ARG INSTALL_MINERU=false

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN if [ "$INSTALL_MINERU" = "true" ]; then pip install --no-cache-dir "mineru[core]"; fi

COPY app ./app
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
