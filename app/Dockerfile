FROM python:3.11-slim

WORKDIR /app

# Installer les dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Copier les requirements et installer
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier l'application
COPY . .

# Exposer le port
EXPOSE 5000

# Variables d'environnement
ENV FLASK_APP=app/app.py
ENV FLASK_ENV=production

# Lancer l'application
CMD ["python", "app/app.py"]
