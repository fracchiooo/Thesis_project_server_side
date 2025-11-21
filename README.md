# Proof of Concept for a Digital Twin of an Ultrasonic Fermentation System - Server part

## Prerequisites

- **GPU Support**: A CUDA-compatible GPU must be available on the host system
- **Docker Environment**: Docker and Docker Compose must be installed and configured
- **TensorFlow Registry Access**: The Docker environment must have access to the TensorFlow public registry (Docker Hub) to download the `tensorflow/tensorflow:2.15.0-gpu` image

## Setup Instructions

### 1. Broker Configuration

- Sign up for the broker cloud console service
- Create a new deployment
- Download the broker's CA certificate
- Rename the certificate to `emqxsl-ca.crt`
- Place the certificate in `./Backend/src/main/resources/certs/`

### 2. Environment Configuration

#### 2.1 Create Environment Variables File

Create a `.env` file in the project root directory with the following key-value pairs:
```env
MODEL_API_KEY=<your_secure_api_key_for_model_microservice>
POSTGRESQL_PASSWORD=<your_database_password>
MQTT_PASSWORD=<password_configured_in_broker_authentication>
JWT_SECRET=<your_random_secure_string>
```


**Note**: For the `MQTT_PASSWORD`, configure the broker's Authentication section with the username set in application.properties, by default is `thesis_yeastime`.

#### 2.2 Configure Application Properties
Edit the file `./Backend/src/main/resources/application.properties` and update the following properties:
```properties
spring.mqtt.host=<your_broker_url>
spring.mqtt.username=<username_configured_in_broker_authentication>
spring.datasource.password=<your_database_password>
```



### 3. Dataset Preparation

- Create the directory `../dataset_conf/`
- Inside this directory, create two CSV files:
  - `dataset.csv` - Initial training dataset
  - `dataset_new_rows.csv` - New observations (initially empty)

#### Dataset Format

The `dataset.csv` file must contain comma-separated values with the following column order:

1. **Duty Cycle**: Value between 0 and 1
2. **Temperature**: Value between 18 and 28 (°C)
3. **Frequency**: Value between 0 and 40 (kHz)
4. **Initial Bacterial Density**: Starting OD value at time 0
5. **Time (t)**: Observation time
6. **Final Bacterial Density**: Observed OD value at time t

### 4. Deployment

Build and start the services:
```bash
docker-compose build
docker-compose up
```

### 5. Access the Application

Once the services are running, the application will be accessible at:
```
http://localhost:80
```

---

**Project**: Proof of Concept for a Digital Twin of an Ultrasonic Fermentation System  
**Institution**: Sapienza University of Rome


