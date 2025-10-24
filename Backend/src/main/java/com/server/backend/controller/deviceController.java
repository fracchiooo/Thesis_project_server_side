package com.server.backend.controller;

import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.util.Date;
import java.util.List;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.domain.Page;
import org.springframework.format.annotation.DateTimeFormat;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import com.server.backend.dto.CommandDto;
import com.server.backend.dto.DeleteDeviceDto;
import com.server.backend.dto.StatusDto;
import com.server.backend.model.Device;
import com.server.backend.model.DeviceStatusLogs;
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

    @PostMapping("/delete")
    public ResponseEntity<Object> deleteDevice(@RequestBody DeleteDeviceDto dto) {
        return deviceServ.deleteDevice(dto.getDevEUI());
    }

    @GetMapping("/list")
    public ResponseEntity<List<Device>> getAllDevices() {
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

    @GetMapping("/getDeviceLogs/{devEUI}")
    public ResponseEntity<List<DeviceStatusLogs>> getDeviceLogs(@PathVariable String devEUI) {
        return deviceServ.getStatussesDevice(devEUI);
    }

    @GetMapping("/getDeviceLogsPages/{devEUI}")
    public Page<DeviceStatusLogs> DeviceStatusLogsPages(@PathVariable String devEUI, 
        @RequestParam @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime start_date,
        @RequestParam @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime finish_date, 
        @RequestParam(defaultValue = "0") int page,
        @RequestParam(defaultValue = "50") int size) {

            Date startDate = Date.from(start_date.toInstant(ZoneOffset.UTC));
            Date finishDate = Date.from(finish_date.toInstant(ZoneOffset.UTC));
            return deviceServ.getLogsByCustomDateRange(devEUI, startDate, finishDate, page, size);
    }
    
    
    
    
}
