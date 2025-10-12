package com.server.backend.controller;

import java.util.List;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.server.backend.dto.CommandDto;
import com.server.backend.dto.StatusDto;
import com.server.backend.service.deviceService;


@RestController
@RequestMapping("/device")
public class deviceController {

    @Autowired
    private deviceService deviceServ;


    @PostMapping("/create")
    public ResponseEntity<StatusDto> createDevice(@RequestBody String devEUI) {        
        return deviceServ.createDevice(devEUI);
    }

    @DeleteMapping("/delete")
    public ResponseEntity<Object> deleteDevice(@RequestBody String devEUI) {
        return deviceServ.deleteDevice(devEUI);
    }


    @GetMapping("/list")
    public ResponseEntity<List<String>> getAllDevices() {
        return deviceServ.getAllDevices();
    }
    
    


    
    @GetMapping("/getStatus/{devEUI}")
    public ResponseEntity<StatusDto> getStatus(@PathVariable String devEUI) {
        return deviceServ.getStatus(devEUI);
    }

    @GetMapping("/getAllStatusses")
    public ResponseEntity<List<StatusDto>> getAllStatusses() {
        return deviceServ.getAllStatusses();
    }

    @PostMapping("sendCommand/{devEUI}")
    public ResponseEntity<Object> sendCommand(@RequestBody CommandDto commandDto, @PathVariable String devEUI) {        
        return deviceServ.sendCommand(commandDto, devEUI);
    }
    
    
}
