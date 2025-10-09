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
    
    public MqttPublisher(MqttClient mqttClient, String publishTopic) {
        this.mqttClient = mqttClient;
    }
    
    public void publish(String topic, String payload) throws MqttException {
        if (!mqttClient.isConnected()) {
            log.warn("MQTT client non connesso, tentativo di riconnessione...");
            mqttClient.reconnect();
        }
        
        MqttMessage message = new MqttMessage(payload.getBytes());
        message.setQos(1);
        message.setRetained(false);
        
        mqttClient.publish(topic, message);
        log.info("Messaggio pubblicato su topic {}: {}", topic, payload);
    }
     
}
