FROM python:3.11-slim

WORKDIR /app

# Dipendenze sistema
RUN apt-get update && apt-get install -y \
    gcc g++ libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Aggiorna pip
RUN pip install --no-cache-dir --upgrade pip

# Installa dipendenze
RUN pip install --no-cache-dir \
    Flask==3.0.0 \
    numpy==1.26.4 \
    tensorflow-cpu==2.15.0 \
    tensorflow-probability==0.23.0 \
    matplotlib==3.8.2

# Copia il codice
COPY . .

# Debug
RUN ls -la /app/ && test -f bayesian_model.py && echo "✓ File found"

EXPOSE 5001

CMD ["python", "bayesian_model.py"]