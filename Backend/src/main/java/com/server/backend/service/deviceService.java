package com.server.backend.service;

import java.util.List;
import java.util.Optional;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.server.backend.dto.commandDto;
import com.server.backend.dto.statusDto;
import com.server.backend.model.Device;
import com.server.backend.model.User;
import com.server.backend.repository.deviceRepository;
import com.server.backend.repository.userRepository;
import com.server.backend.utilities.JWTContext;

//TODO add script to subscribe all the current active devices to their relative queues


@Service
public class deviceService {

    
    @Autowired
    private ObjectMapper objectMapper;

    @Autowired
    private deviceRepository deviceRepo;

    @Autowired
    private userRepository userRepo;

    public ResponseEntity<statusDto> getStatus(String devEUI) {
        Optional<Device> dev = deviceRepo.findById(devEUI);

        if(dev.isEmpty()){
            return ResponseEntity.notFound().build();
        }

        Device device = dev.get();
        statusDto status = new statusDto();
        status.setDeviceEUI(device.getDeviceEUI());
        status.setLastUpdate(device.getLastUpdate());
        status.setUsername(device.getUser().getUsername());   
        status.setDeviceEnvRequests(device.getDeviceEnvRequests());
        return ResponseEntity.ok(status);
    }

    public ResponseEntity<List<statusDto>> getAllStatusses() {
        List<Device> devices = deviceRepo.findAll();

        if(devices.isEmpty()){
            return ResponseEntity.noContent().build();
        }

        List<statusDto> statuses = devices.stream().map(device -> {
            statusDto status = objectMapper.convertValue(device, statusDto.class);
            status.setUsername(device.getUser().getUsername());   
            return status;
        }).toList();
        return ResponseEntity.ok(statuses);
    }


    public ResponseEntity<Object> sendCommand(commandDto commandDto) {
        return null;
        //TODO subscribe to mqtt class
    }     

    
    public ResponseEntity<statusDto> createDevice(String devEUI) {   
        User u = userRepo.findOneByUsername(JWTContext.get()).orElse(null);    
        Device d = new Device(devEUI, u);
        d = deviceRepo.save(d);
        deviceRepo.flush();
        statusDto status = objectMapper.convertValue(d, statusDto.class);
        status.setUsername(d.getUser().getUsername());   
        return ResponseEntity.ok(status);
        
        //TODO subscribe to mqtt class

    }

    public ResponseEntity<Object> deleteDevice(String devEUI) {
        return deviceRepo.findById(devEUI).map(device -> {
            deviceRepo.delete(device);
            deviceRepo.flush();
            return ResponseEntity.ok().build();
        }).orElse(ResponseEntity.notFound().build());

        // TODO unsubscribe from mqtt
    }


    public ResponseEntity<List<String>> getAllDevices() {
        List<String> devices = deviceRepo.findAll().stream().map(d -> d.getDeviceEUI()).toList();
        if(devices.isEmpty()){
            return ResponseEntity.noContent().build();
        }
        return ResponseEntity.ok(devices);
    }

    
}
