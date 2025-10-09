FROM python:3.11-slim

WORKDIR /app

# Copia i requirements e installa le dipendenze
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia il codice dell'applicazione
COPY app.py .

# Esponi la porta
EXPOSE 5000

# Comando per avviare l'applicazione
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "app:app"]