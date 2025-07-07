FROM python:3.10-slim

WORKDIR /app

# Instala dependencias del sistema, incluyendo libGL
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY . /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

RUN python gfpgan_download.py

EXPOSE 5000

CMD ["python", "app.py"]
