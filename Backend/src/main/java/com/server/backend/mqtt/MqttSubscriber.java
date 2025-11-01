package com.server.backend.mqtt;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

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
public class MqttSubscriber implements MqttCallback {

    private final MqttClient mqttClient;
    private final ApplicationEventPublisher eventPublisher;
    
    private final Map<String, Integer> activeSubscriptions = new ConcurrentHashMap<>();
    
    public MqttSubscriber(MqttClient mqttClient, ApplicationEventPublisher eventPublisher) {
        this.mqttClient = mqttClient;
        this.eventPublisher = eventPublisher;
    }
    
    @PostConstruct
    private void initialize() {
        mqttClient.setCallback(this);
        log.info("MqttSubscriber initialized");
    }
    
    public void subscribe(String topic) throws MqttException {
        subscribe(topic, 1);
    }
    
    public synchronized void subscribe(String topic, int qos) throws MqttException {
        if (activeSubscriptions.containsKey(topic)) {
            log.warn("Already subscribed to the topic '{}', updated QoS from {} to {}", 
                     topic, activeSubscriptions.get(topic), qos);
        }
        
        mqttClient.subscribe(topic, qos);
        activeSubscriptions.put(topic, qos);
        log.info("Subscribed to topic '{}' with QoS {}", topic, qos);
    }
    
    public synchronized void subscribeMultiple(Map<String, Integer> topicsWithQos) throws MqttException {
        String[] topics = topicsWithQos.keySet().toArray(new String[0]);
        int[] qosLevels = topicsWithQos.values().stream().mapToInt(Integer::intValue).toArray();
        
        mqttClient.subscribe(topics, qosLevels);
        activeSubscriptions.putAll(topicsWithQos);
        
        log.info("Subscribed to {} topic: {}", topics.length, String.join(", ", topics));
    }
    
    public synchronized void subscribeMultiple(String[] topics, int qos) throws MqttException {
        int[] qosLevels = new int[topics.length];
        for (int i = 0; i < topics.length; i++) {
            qosLevels[i] = qos;
        }
        
        mqttClient.subscribe(topics, qosLevels);
        
        for (String topic : topics) {
            activeSubscriptions.put(topic, qos);
        }
        
        log.info("Subscribed to {} topic with QoS {}: {}", 
                 topics.length, qos, String.join(", ", topics));
    }
    
    public synchronized void unsubscribe(String topic) throws MqttException {
        if (!activeSubscriptions.containsKey(topic)) {
            log.warn("Attempt to unsubscribe to a topic '{}' alredy unsubscribed", topic);
            return;
        }
        
        mqttClient.unsubscribe(topic);
        activeSubscriptions.remove(topic);
        log.info("Unsubscribed to the topic '{}'", topic);
    }
    
    public synchronized void unsubscribeMultiple(String[] topics) throws MqttException {
        mqttClient.unsubscribe(topics);
        
        for (String topic : topics) {
            activeSubscriptions.remove(topic);
        }
        
        log.info("Unsubscribed from {} topic: {}", topics.length, String.join(", ", topics));
    }
    
    public synchronized void unsubscribeAll() throws MqttException {
        if (activeSubscriptions.isEmpty()) {
            log.info("No active subscription removable");
            return;
        }
        
        String[] topics = activeSubscriptions.keySet().toArray(new String[0]);
        mqttClient.unsubscribe(topics);
        
        int count = activeSubscriptions.size();
        activeSubscriptions.clear();
        log.info("Unsubscribed from all the {} active topics", count);
    }
    
    public Map<String, Integer> getActiveSubscriptions() {
        return new ConcurrentHashMap<>(activeSubscriptions);
    }
    
    public boolean isSubscribed(String topic) {
        return activeSubscriptions.containsKey(topic);
    }
    

    private synchronized void reconnectAndResubscribe() {
        if (activeSubscriptions.isEmpty()) {
            log.info("No subscription to restore");
            return;
        }
        
        try {
            log.info("Restore of {} subscriptions...", activeSubscriptions.size());
            
            String[] topics = activeSubscriptions.keySet().toArray(new String[0]);
            int[] qosLevels = activeSubscriptions.values().stream()
                                                  .mapToInt(Integer::intValue)
                                                  .toArray();
            
            mqttClient.subscribe(topics, qosLevels);
            
            log.info("{} restored subscriptions: {}", 
                     topics.length, String.join(", ", topics));
            
        } catch (MqttException e) {
            log.error("Error during subscription's restore: {}", e.getMessage(), e);
            
            // Retry dopo breve attesa
            try {
                Thread.sleep(500);
                log.info("Second try on restoring the subscriptions...");
                
                String[] topics = activeSubscriptions.keySet().toArray(new String[0]);
                int[] qosLevels = activeSubscriptions.values().stream()
                                                      .mapToInt(Integer::intValue)
                                                      .toArray();
                mqttClient.subscribe(topics, qosLevels);
                log.info("Subscription restored");
                
            } catch (Exception retryException) {
                log.error("Definitely impossible to restore the subscriptions", retryException);
            }
        }
    }
    
    
    @Override
    public void connectionLost(Throwable cause) {
        log.error("MQTT connection lost: {} - {}", 
                 cause.getClass().getSimpleName(), 
                 cause.getMessage());
        scheduleResubscribe();
    }
    
   
    private void scheduleResubscribe() {
        new Thread(() -> {
            try {
                int attempts = 0;
                int maxAttempts = 20; // 20 seconds max
                
                while (!mqttClient.isConnected() && attempts < maxAttempts) {
                    attempts++;
                    log.debug("Waiting reconenction... {}/{}", attempts, maxAttempts);
                    Thread.sleep(1000);
                }
                
                if (mqttClient.isConnected()) {
                    log.info("Client reconnected, restoring subscriptions...");
                    
                    Thread.sleep(100);
                    reconnectAndResubscribe();
                } else {
                    log.error("Timeout reconnections after {} seconds", maxAttempts);
                }
                
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                log.error("Thread for restoring subscriptions interrupted", e);
            }
        }, "MQTT-Resubscribe-Thread").start();
    }
    
    @Override
    public void messageArrived(String topic, MqttMessage message) throws Exception {
        String payload = new String(message.getPayload());
        
        log.info("MESSAGE - Topic: '{}', QoS: {}, Retained: {}, Size: {} bytes", 
                 topic, message.getQos(), message.isRetained(), payload.length());
        
        if (payload.length() <= 200) {
            log.debug("Payload: {}", payload);
        } else {
            log.debug("Payload (truncated, because too long to log): {}...", payload.substring(0, 200));
        }
        
        MqttMessageEvent event = new MqttMessageEvent(this, topic, payload);
        eventPublisher.publishEvent(event);
    }
    
    @Override
    public void deliveryComplete(IMqttDeliveryToken token) {
        try {
            log.debug("✓ Delivery complete - MessageId: {}", token.getMessageId());
        } catch (Exception e) {
            log.debug("✓ Delivery complete");
        }
    }
}