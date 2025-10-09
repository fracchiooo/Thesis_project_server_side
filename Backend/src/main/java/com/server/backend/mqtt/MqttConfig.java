package com.server.backend.mqtt;

import org.eclipse.paho.client.mqttv3.MqttClient;
import org.eclipse.paho.client.mqttv3.MqttConnectOptions;
import org.eclipse.paho.client.mqttv3.MqttException;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class MqttConfig {
   
   @Value("${spring.mqtt.host}")
   protected String broker;

   @Value("${spring.mqtt.port}")
   protected Integer port;

    @Value("${spring.mqtt.client_id}")
    private String clientId;



    @Bean
    public MqttClient mqttClient() throws MqttException {
        String broker_url = String.format("tcp://%s:%d", broker, port);
        MqttClient client = new MqttClient(broker_url, clientId);
        MqttConnectOptions options = new MqttConnectOptions();
        options.setCleanSession(true);
        options.setAutomaticReconnect(true);
        options.setConnectionTimeout(10);
        
        client.connect(options);
        return client;
    }
   

}