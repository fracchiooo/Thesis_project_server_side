package com.server.backend.mqtt;

import org.eclipse.paho.client.mqttv3.IMqttDeliveryToken;
import org.eclipse.paho.client.mqttv3.MqttCallback;
import org.eclipse.paho.client.mqttv3.MqttClient;
import org.eclipse.paho.client.mqttv3.MqttException;
import org.eclipse.paho.client.mqttv3.MqttMessage;
import org.springframework.context.ApplicationEventPublisher;
import org.springframework.stereotype.Component;

import jakarta.annotation.PostConstruct;
import lombok.extern.slf4j.Slf4j;

@Component
@Slf4j
public class MqttSubscriber implements MqttCallback{

    private final MqttClient mqttClient;
    private final ApplicationEventPublisher eventPublisher;
    private final String subscribeTopic;
    
    public MqttSubscriber(MqttClient mqttClient, ApplicationEventPublisher eventPublisher, String subscribeTopic) {
        this.mqttClient = mqttClient;
        this.eventPublisher = eventPublisher;
        this.subscribeTopic = subscribeTopic;
        initializeSubscription();
    }
    
    @PostConstruct
    private void initializeSubscription() {
        try {
            mqttClient.setCallback(this);
            mqttClient.subscribe(subscribeTopic, 1);
            log.info("Sottoscritto al topic: {}", subscribeTopic);
        } catch (MqttException e) {
            log.error("Errore durante la sottoscrizione al topic", e);
        }
    }
    
    public void update_subscribe(String topic, int qos) throws MqttException {
        mqttClient.subscribe(topic, qos);
        log.info("Sottoscritto al topic: {}", topic);
    }
    
    @Override
    public void connectionLost(Throwable cause) {
        log.error("Connessione MQTT persa", cause);
        // Puoi implementare logica di riconnessione automatica
    }
    
    @Override
    public void messageArrived(String topic, MqttMessage message) throws Exception {
        String payload = new String(message.getPayload());
        log.info("Messaggio ricevuto dal topic {}: {}", topic, payload);
        
        // Pubblica un evento Spring per permettere ai service di gestirlo
        MqttMessageEvent event = new MqttMessageEvent(this, topic, payload);
        eventPublisher.publishEvent(event);
    }
    
    @Override
    public void deliveryComplete(IMqttDeliveryToken token) {
        log.debug("Delivery complete: {}", token);
    }
    
}
