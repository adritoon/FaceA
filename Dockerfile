# Usa Python 3.10
FROM python:3.10-slim

# Establece el directorio de trabajo
WORKDIR /app

# Copia archivos
COPY . /app

# Instala dependencias
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Descarga los modelos
RUN python gfpgan_download.py

# Expone el puerto
EXPOSE 5000

# Comando para iniciar la app
CMD ["python", "app.py"]
