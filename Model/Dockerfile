FROM tensorflow/tensorflow:2.15.0-gpu
WORKDIR /app
RUN apt-get update && apt-get install -y gcc g++ libgomp1 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir --no-deps Flask Werkzeug Jinja2 itsdangerous click MarkupSafe
RUN pip install --no-cache-dir --ignore-installed blinker
RUN pip install --no-cache-dir tensorflow-probability==0.23.0 matplotlib
COPY . .
EXPOSE 5001
ENV TF_FORCE_GPU_ALLOW_GROWTH=true
CMD ["python", "bayesian_model.py"]