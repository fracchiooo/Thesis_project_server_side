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
        log.info("MqttSubscriber inizializzato e pronto per sottoscrizioni");
    }
    
    public void subscribe(String topic) throws MqttException {
        subscribe(topic, 1);
    }
    
    public synchronized void subscribe(String topic, int qos) throws MqttException {
        if (activeSubscriptions.containsKey(topic)) {
            log.warn("Già sottoscritto al topic '{}', aggiornamento QoS da {} a {}", 
                     topic, activeSubscriptions.get(topic), qos);
        }
        
        mqttClient.subscribe(topic, qos);
        activeSubscriptions.put(topic, qos);
        log.info("✓ Sottoscritto al topic '{}' con QoS {}", topic, qos);
    }
    
    public synchronized void subscribeMultiple(Map<String, Integer> topicsWithQos) throws MqttException {
        String[] topics = topicsWithQos.keySet().toArray(new String[0]);
        int[] qosLevels = topicsWithQos.values().stream().mapToInt(Integer::intValue).toArray();
        
        mqttClient.subscribe(topics, qosLevels);
        activeSubscriptions.putAll(topicsWithQos);
        
        log.info("✓ Sottoscritto a {} topic: {}", topics.length, String.join(", ", topics));
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
        
        log.info("✓ Sottoscritto a {} topic con QoS {}: {}", 
                 topics.length, qos, String.join(", ", topics));
    }
    
    public synchronized void unsubscribe(String topic) throws MqttException {
        if (!activeSubscriptions.containsKey(topic)) {
            log.warn("Tentativo di disiscrizione da topic '{}' non sottoscritto", topic);
            return;
        }
        
        mqttClient.unsubscribe(topic);
        activeSubscriptions.remove(topic);
        log.info("✗ Disiscritto dal topic '{}'", topic);
    }
    
    public synchronized void unsubscribeMultiple(String[] topics) throws MqttException {
        mqttClient.unsubscribe(topics);
        
        for (String topic : topics) {
            activeSubscriptions.remove(topic);
        }
        
        log.info("✗ Disiscritto da {} topic: {}", topics.length, String.join(", ", topics));
    }
    
    public synchronized void unsubscribeAll() throws MqttException {
        if (activeSubscriptions.isEmpty()) {
            log.info("Nessuna sottoscrizione attiva da rimuovere");
            return;
        }
        
        String[] topics = activeSubscriptions.keySet().toArray(new String[0]);
        mqttClient.unsubscribe(topics);
        
        int count = activeSubscriptions.size();
        activeSubscriptions.clear();
        log.info("✗ Disiscritto da tutti i {} topic attivi", count);
    }
    
    public Map<String, Integer> getActiveSubscriptions() {
        return new ConcurrentHashMap<>(activeSubscriptions);
    }
    
    public boolean isSubscribed(String topic) {
        return activeSubscriptions.containsKey(topic);
    }
    
    /**
     * CORRETTO: Ripristina sottoscrizioni IMMEDIATAMENTE dopo riconnessione
     */
    private synchronized void reconnectAndResubscribe() {
        if (activeSubscriptions.isEmpty()) {
            log.info("Nessuna sottoscrizione da ripristinare");
            return;
        }
        
        try {
            log.info("⚡ Ripristino IMMEDIATO di {} sottoscrizioni...", activeSubscriptions.size());
            
            String[] topics = activeSubscriptions.keySet().toArray(new String[0]);
            int[] qosLevels = activeSubscriptions.values().stream()
                                                  .mapToInt(Integer::intValue)
                                                  .toArray();
            
            // CRITICO: Sottoscrivi IMMEDIATAMENTE
            mqttClient.subscribe(topics, qosLevels);
            
            log.info("✓ {} sottoscrizioni ripristinate: {}", 
                     topics.length, String.join(", ", topics));
            
        } catch (MqttException e) {
            log.error("❌ ERRORE critico durante ripristino sottoscrizioni: {}", e.getMessage(), e);
            
            // Retry dopo breve attesa
            try {
                Thread.sleep(500);
                log.info("Secondo tentativo di ripristino sottoscrizioni...");
                
                String[] topics = activeSubscriptions.keySet().toArray(new String[0]);
                int[] qosLevels = activeSubscriptions.values().stream()
                                                      .mapToInt(Integer::intValue)
                                                      .toArray();
                mqttClient.subscribe(topics, qosLevels);
                log.info("✓ Sottoscrizioni ripristinate al secondo tentativo");
                
            } catch (Exception retryException) {
                log.error("❌ Impossibile ripristinare sottoscrizioni dopo retry", retryException);
            }
        }
    }
    
    // ========== Callback MqttCallback ==========
    
    @Override
    public void connectionLost(Throwable cause) {
        log.error("⚠️ Connessione MQTT persa: {} - {}", 
                 cause.getClass().getSimpleName(), 
                 cause.getMessage());
        
        if (cause instanceof java.io.EOFException) {
            log.error("🔴 EOFException rilevata - probabilmente problema con sottoscrizioni o sessione");
        }
        
        // La riconnessione automatica è gestita da Paho
        // Quando si riconnette, chiamiamo reconnectAndResubscribe()
        scheduleResubscribe();
    }
    
    /**
     * Schedula il ripristino delle sottoscrizioni dopo la riconnessione
     */
    private void scheduleResubscribe() {
        new Thread(() -> {
            try {
                // Attendi che la riconnessione sia completa
                int attempts = 0;
                int maxAttempts = 20; // 20 secondi max
                
                while (!mqttClient.isConnected() && attempts < maxAttempts) {
                    attempts++;
                    log.debug("Attesa riconnessione... {}/{}", attempts, maxAttempts);
                    Thread.sleep(1000);
                }
                
                if (mqttClient.isConnected()) {
                    log.info("✓ Client riconnesso, ripristino sottoscrizioni...");
                    
                    // CRITICO: Piccola pausa per stabilizzare la connessione
                    Thread.sleep(100);
                    
                    // Ripristina sottoscrizioni
                    reconnectAndResubscribe();
                } else {
                    log.error("❌ Timeout riconnessione dopo {} secondi", maxAttempts);
                }
                
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                log.error("Thread ripristino sottoscrizioni interrotto", e);
            }
        }, "MQTT-Resubscribe-Thread").start();
    }
    
    @Override
    public void messageArrived(String topic, MqttMessage message) throws Exception {
        String payload = new String(message.getPayload());
        
        log.info("📩 Messaggio - Topic: '{}', QoS: {}, Retained: {}, Size: {} bytes", 
                 topic, message.getQos(), message.isRetained(), payload.length());
        
        if (payload.length() <= 200) {
            log.debug("Payload: {}", payload);
        } else {
            log.debug("Payload (troncato): {}...", payload.substring(0, 200));
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