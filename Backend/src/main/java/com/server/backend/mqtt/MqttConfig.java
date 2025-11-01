package com.server.backend.mqtt;

import java.io.InputStream;
import java.security.KeyStore;
import java.security.cert.CertificateFactory;
import java.security.cert.X509Certificate;

import javax.net.ssl.SSLContext;
import javax.net.ssl.SSLSocketFactory;
import javax.net.ssl.TrustManagerFactory;

import org.eclipse.paho.client.mqttv3.MqttClient;
import org.eclipse.paho.client.mqttv3.MqttConnectOptions;
import org.eclipse.paho.client.mqttv3.MqttException;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.core.io.Resource;

import jakarta.annotation.PreDestroy;
import lombok.extern.slf4j.Slf4j;

@Configuration
@Slf4j
public class MqttConfig {
   
   @Value("${spring.mqtt.host}")
   protected String broker;

   @Value("${spring.mqtt.port}")
   protected Integer port;

    @Value("${spring.mqtt.client_id}")
    private String clientId;

    @Value("${spring.mqtt.username}")
    private String username;
    
    @Value("${spring.mqtt.password}")
    private String password;
    
    @Value("${spring.mqtt.tls.enabled:true}")
    private boolean tlsEnabled;

    @Value("${spring.mqtt.tls.ca-cert:#{null}}")
    private Resource caCertResource;

    private MqttClient client;


    @Bean
    public MqttClient mqttClient() throws MqttException {
        // Uses ssl:// if TLS enabled (it is by default), otherwise tcp://
        String protocol = tlsEnabled ? "ssl" : "tcp";
        String brokerUrl = String.format("%s://%s:%d", protocol, broker, port);
        
        log.info("Connecting to a MQTT broker: {}", brokerUrl);
        
        client = new MqttClient(brokerUrl, clientId);
        MqttConnectOptions options = new MqttConnectOptions();
        
        options.setCleanSession(false);
        options.setAutomaticReconnect(true);
        options.setConnectionTimeout(30);
        options.setKeepAliveInterval(60);
        options.setMaxReconnectDelay(5000);

        // Username and password auth
        if (username != null && !username.isEmpty()) {
            options.setUserName(username);
            log.info("Username configurato: {}", username);
        }
        if (password != null && !password.isEmpty()) {
            options.setPassword(password.toCharArray());
            log.info("Password configurata");
        }
        
        // TLS configuration
        if (tlsEnabled) {
            try {
                SSLSocketFactory socketFactory = getSSLSocketFactory();
                options.setSocketFactory(socketFactory);
                log.info("TLS/SSL configurato con successo");
            } catch (Exception e) {
                log.error("Errore durante la configurazione TLS", e);
                throw new MqttException(e);
            }
        }
        
        connectWithRetry(options, 3);
        
        return client;
    }

    private void connectWithRetry(MqttConnectOptions options, int maxAttempts) throws MqttException {
        int attempt = 0;
        MqttException lastException = null;
        
        while (attempt < maxAttempts) {
            try {
                attempt++;
                log.info("Connection attempt {}/{}", attempt, maxAttempts);
                client.connect(options);
                log.info("Successfully connected to a broker MQTT");
                return;
            } catch (MqttException e) {
                lastException = e;
                log.warn("Attempt {} failed: {}", attempt, e.getMessage());
                
                if (attempt < maxAttempts) {
                    try {
                        Thread.sleep(2000 * attempt); // Exponential backoff
                    } catch (InterruptedException ie) {
                        Thread.currentThread().interrupt();
                        throw new MqttException(ie);
                    }
                }
            }
        }
        throw new MqttException(lastException);
    }
    
    /**
     * Creates SSLSocketFactory from the broker certificate CA
     */
    private SSLSocketFactory getSSLSocketFactory() throws Exception {
        if (caCertResource == null) {
            log.info("No certificate CA specified, using system's ones");
            SSLContext sslContext = SSLContext.getInstance("TLS");
            sslContext.init(null, null, null);
            return sslContext.getSocketFactory();
        }
        
        log.info("Loaded certificate CA from: {}", caCertResource.getDescription());
        
        CertificateFactory cf = CertificateFactory.getInstance("X.509");
        X509Certificate caCert;
        
        try (InputStream inputStream = caCertResource.getInputStream()) {
            caCert = (X509Certificate) cf.generateCertificate(inputStream);
        }
        
        KeyStore keyStore = KeyStore.getInstance(KeyStore.getDefaultType());
        keyStore.load(null, null);
        keyStore.setCertificateEntry("ca-certificate", caCert);
        
        TrustManagerFactory tmf = TrustManagerFactory.getInstance(
            TrustManagerFactory.getDefaultAlgorithm()
        );
        tmf.init(keyStore);
        
        SSLContext sslContext = SSLContext.getInstance("TLS");
        sslContext.init(null, tmf.getTrustManagers(), null);
        
        log.info("Certificate CA successfully loaded");
        return sslContext.getSocketFactory();
    }


    @PreDestroy
    public void shutdown() {
        try {
            if (client != null && client.isConnected()) {
                log.info("Disconnecting from broker MQTT...");
                client.disconnect(5000);
            }
            if (client != null) {
                client.close();
            }
            log.info("Client MQTT correctly closed");
        } catch (MqttException e) {
            log.error("Error disconnecting MQTT client", e);
        }
    }
}
   