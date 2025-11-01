package com.server.backend.utilities;

import java.util.Collections;
import java.util.List;
import java.util.Optional;

import org.eclipse.paho.client.mqttv3.MqttException;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.CommandLineRunner;
import org.springframework.stereotype.Component;

import com.server.backend.model.Device;
import com.server.backend.mqtt.MqttSubscriber;
import com.server.backend.service.DeviceService;

import lombok.extern.slf4j.Slf4j;

@Component
@Slf4j
public class AppStartupRunner implements CommandLineRunner {

    @Autowired
    private MqttSubscriber mqttSubscriber;

    @Autowired
    private DeviceService deviceService;

    @Override
    public void run(String... args) throws Exception {
        
        log.info("Server started - Executing MQTT subscriptions per the devices registered in the database...");

        List<String> allDeviceEUIs = Optional.ofNullable(deviceService.getAllDevices().getBody())
            .orElse(Collections.emptyList())
            .stream()
            .map(Device::getDeviceEUI)
            .toList();
                
        if (allDeviceEUIs == null || allDeviceEUIs.isEmpty()) {
            log.info("No device found for the startup subscriptions.");
            return;
        }
        for (String devEUI : allDeviceEUIs) {
            try {
                mqttSubscriber.subscribe(devEUI+"/uplink");
            } catch (MqttException e) {
                log.error("Error during the subscription of the topic for the device {}: {}", devEUI, e.getMessage());
            }
        }
        
        log.info("Startup subscriptions completed.");
    }
}
