package com.server.backend.service;

import java.util.List;
import java.util.Optional;

import org.eclipse.paho.client.mqttv3.MqttException;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.event.EventListener;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.server.backend.dto.CommandDto;
import com.server.backend.dto.MessageUplinkDto;
import com.server.backend.dto.StatusDto;
import com.server.backend.model.Device;
import com.server.backend.model.User;
import com.server.backend.mqtt.MqttMessageEvent;
import com.server.backend.mqtt.MqttPublisher;
import com.server.backend.mqtt.MqttSubscriber;
import com.server.backend.repository.deviceRepository;
import com.server.backend.repository.userRepository;
import com.server.backend.utilities.JWTContext;

import jakarta.persistence.EntityManager;

//TODO add script to subscribe all the current active devices to their relative queues


@Service
public class deviceService {

    
    @Autowired
    private ObjectMapper objectMapper;

    @Autowired
    private deviceRepository deviceRepo;

    @Autowired
    private userRepository userRepo;

    @Autowired
    private MqttSubscriber mqttSubscriber;

    @Autowired
    private MqttPublisher mqttPublisher;

    @Autowired
    private EntityManager entityManager;


    public ResponseEntity<StatusDto> getStatus(String devEUI) {
        Optional<Device> dev = deviceRepo.findById(devEUI);

        if(dev.isEmpty()){
            return ResponseEntity.notFound().build();
        }

        Device device = dev.get();
        StatusDto status = new StatusDto();
        status.setDeviceEUI(device.getDeviceEUI());
        status.setLastUpdate(device.getLastUpdate());
        status.setUsername(device.getUser().getUsername());   
        status.setDeviceEnvRequests(device.getDeviceEnvRequests());
        return ResponseEntity.ok(status);
    }

    public ResponseEntity<List<StatusDto>> getAllStatusses() {
        List<Device> devices = deviceRepo.findAll();

        if(devices.isEmpty()){
            return ResponseEntity.noContent().build();
        }

        List<StatusDto> statuses = devices.stream().map(device -> {
            StatusDto status = objectMapper.convertValue(device, StatusDto.class);
            status.setUsername(device.getUser().getUsername());   
            return status;
        }).toList();
        return ResponseEntity.ok(statuses);
    }


    public ResponseEntity<Object> sendCommand(CommandDto commandDto, String devEUI) {

        Optional<Device> dev = deviceRepo.findById(devEUI);
        if(dev.isEmpty()){
            return ResponseEntity.notFound().build();
        }

        try{
            mqttPublisher.publish(devEUI+"/downlink", objectMapper.writeValueAsString(commandDto));
            return ResponseEntity.ok().build();
        } catch (MqttException e){
            System.out.println("Error in sending the command to the device: " + devEUI);
            return ResponseEntity.status(500).body("Error in sending the command to the device");
        } catch(JsonProcessingException e){
            System.out.println("Error in converting the command to JSON string for device: " + devEUI);
            return ResponseEntity.status(500).body("Error in converting the command to JSON string");
        }
    }     

    
    public ResponseEntity<StatusDto> createDevice(String devEUI) {   
        User u = userRepo.findOneByUsername(JWTContext.get()).orElse(null);
        if(devEUI == null || devEUI.isEmpty() || u == null || deviceRepo.existsById(devEUI)){
            return ResponseEntity.status(400).body(null);
        }
        Device d = new Device(devEUI, u);
        d = deviceRepo.save(d);
        deviceRepo.flush();
        StatusDto status = objectMapper.convertValue(d, StatusDto.class);
        status.setUsername(d.getUser().getUsername()); 
        
        System.out.println("Created device: " + d.getDeviceEUI() + " for user: " + d.getUser().getUsername());
        System.out.println("Subscribing to the topic for the device: " + d.getDeviceEUI());
        try{
            mqttSubscriber.subscribe(devEUI+"/uplink");
        } catch (MqttException e){
            System.out.println("Error subscribing to the topic for the device: " + devEUI+ " ,the subscription will be retried later");
        }

        return ResponseEntity.ok(status);
    }

    public ResponseEntity<Object> deleteDevice(String devEUI) {
        try{
            mqttSubscriber.unsubscribe(devEUI);
        } catch (MqttException e){
            System.out.println("Failed in unsubscribing the device: " + devEUI + " from its topic");
        }

        return deviceRepo.findById(devEUI).map(device -> {
            deviceRepo.delete(device);
            deviceRepo.flush();
            return ResponseEntity.ok().build();
        }).orElse(ResponseEntity.notFound().build());

    }


    public ResponseEntity<List<String>> getAllDevices() {
        List<String> devices = deviceRepo.findAll().stream().map(d -> d.getDeviceEUI()).toList();
        if(devices.isEmpty()){
            return ResponseEntity.noContent().build();
        }
        return ResponseEntity.ok(devices);
    }

    @EventListener
    @Transactional
    public void handleMqttMessage(MqttMessageEvent event) {
        String topic = event.getTopic();
        String payload = event.getPayload();
        System.out.println("Handling MQTT message for topic: " + topic);

        Optional<Device> devOpt = deviceRepo.findById(topic.replace("/uplink", ""));
        if(devOpt.isEmpty()){
            System.out.println("Received message for unknown device: " + topic);
            return;
        }
        Device device = devOpt.get();
        System.out.println(payload);
        try {
            // Assuming payload is a JSON string with fields matching Device's fields
            MessageUplinkDto message = objectMapper.readValue(payload, MessageUplinkDto.class);
            device.setLastUpdate(message.getLastUpdate());
            device.setCurrentTemperature(message.getCurrentTemperature());

            device.setDeviceEnvRequests("duty_frequency", message.getDeviceEnvRequests().getDuty_frequency());
            device.setDeviceEnvRequests("frequency", message.getDeviceEnvRequests().getFrequency());
            device.setDeviceEnvRequests("finish after", message.getDeviceEnvRequests().getFinishAfter());
            device.setDeviceEnvRequests("start time", message.getDeviceEnvRequests().getStartTime());
            device.setDeviceEnvRequests("temperature", message.getDeviceEnvRequests().getTemperature());
            
            deviceRepo.save(device);
            deviceRepo.flush();
            entityManager.flush();

            System.out.println("Updated device status for: " + topic);
            //get the data from dto and update the device status
        } catch (Exception e) {
            System.out.println("❌ Parsing failed: "+e.getMessage());
            System.out.println("Payload was: "+ payload);
            System.out.println("Error parsing JSON payload for device: " + topic);
        }
    }
}
