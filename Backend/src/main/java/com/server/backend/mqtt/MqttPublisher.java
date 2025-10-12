package com.server.backend.mqtt;

import org.eclipse.paho.client.mqttv3.MqttClient;
import org.eclipse.paho.client.mqttv3.MqttException;
import org.eclipse.paho.client.mqttv3.MqttMessage;
import org.springframework.stereotype.Component;

import lombok.extern.slf4j.Slf4j;

@Component
@Slf4j
public class MqttPublisher {

    private final MqttClient mqttClient;
    
    public MqttPublisher(MqttClient mqttClient) {
        this.mqttClient = mqttClient;
    }
    
    public void publish(String topic, String payload) throws MqttException {
        publish(topic, payload, 1, false);
    }
    
    public void publish(String topic, String payload, int qos, boolean retained) throws MqttException {
        // NON chiamare reconnect() manualmente - lascia gestire alla riconnessione automatica
        if (!mqttClient.isConnected()) {
            log.warn("MQTT client non connesso, messaggio non inviato. Riconnessione automatica in corso...");
            throw new MqttException(MqttException.REASON_CODE_CLIENT_NOT_CONNECTED);
        }
        
        MqttMessage message = new MqttMessage(payload.getBytes());
        message.setQos(qos);
        message.setRetained(retained);
        
        mqttClient.publish(topic, message);
        log.info("✓ Messaggio pubblicato su topic '{}' (QoS {}): {}", topic, qos, 
                 payload.length() > 100 ? payload.substring(0, 100) + "..." : payload);
    }
    
    // Metodo alternativo con retry
    public void publishWithRetry(String topic, String payload, int qos, boolean retained, int maxRetries) {
        int attempt = 0;
        
        while (attempt < maxRetries) {
            try {
                publish(topic, payload, qos, retained);
                return;
            } catch (MqttException e) {
                attempt++;
                log.warn("Tentativo di pubblicazione {}/{} fallito: {}", attempt, maxRetries, e.getMessage());
                
                if (attempt < maxRetries) {
                    try {
                        Thread.sleep(1000 * attempt);
                    } catch (InterruptedException ie) {
                        Thread.currentThread().interrupt();
                        break;
                    }
                }
            }
        }
        
        log.error("✗ Impossibile pubblicare il messaggio dopo {} tentativi", maxRetries);
    }
     
}
