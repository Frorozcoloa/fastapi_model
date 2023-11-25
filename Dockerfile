# Usa la imagen oficial de Python
FROM python:3.10.0-slim-buster

# Actualiza el sistema y instala las dependencias
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get upgrade -y


# Establece el directorio de trabajo en /app
WORKDIR /app

# Copia el archivo de requisitos en el contenedor
COPY requirements.txt .

# Instala las dependencias
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copia los archivos necesarios al contenedor
COPY . .

# Expone el puerto 8000 en el contenedor
EXPOSE 8000

# Comando para ejecutar la aplicación cuando se inicie el contenedor
ENTRYPOINT [ "./entrypoint.sh" ]
