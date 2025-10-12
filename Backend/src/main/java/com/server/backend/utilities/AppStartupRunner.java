package com.server.backend.utilities;

import java.util.List;

import org.eclipse.paho.client.mqttv3.MqttException;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.CommandLineRunner;
import org.springframework.stereotype.Component;

import com.server.backend.mqtt.MqttSubscriber;
import com.server.backend.service.deviceService;

import lombok.extern.slf4j.Slf4j;

@Component
@Slf4j
public class AppStartupRunner implements CommandLineRunner {

    @Autowired
    private MqttSubscriber mqttSubscriber;

    @Autowired
    private deviceService deviceService;

    @Override
    public void run(String... args) throws Exception {
        
        log.info("✓ Applicazione avviata - Eseguo sottoscrizioni MQTT per i dispositivi esistenti...");

        List<String> allDeviceEUIs = deviceService.getAllDevices().getBody();
        if (allDeviceEUIs == null || allDeviceEUIs.isEmpty()) {
            log.info("Nessun dispositivo trovato per la sottoscrizione.");
            return;
        }
        for (String devEUI : allDeviceEUIs) {
            try {
                mqttSubscriber.subscribe(devEUI+"/uplink");
            } catch (MqttException e) {
                log.error("Errore durante la sottoscrizione al topic del dispositivo {}: {}", devEUI, e.getMessage());
            }
        }
        
        log.info("✓ Sottoscrizioni iniziali completate.");
    }
    
}
