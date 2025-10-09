package com.server.backend.model;

import java.util.Date;
import java.util.HashMap;
import java.util.Map;

import com.fasterxml.jackson.annotation.JsonIgnore;

import jakarta.persistence.CollectionTable;
import jakarta.persistence.Column;
import jakarta.persistence.ElementCollection;
import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.JoinColumn;
import jakarta.persistence.ManyToOne;
import jakarta.persistence.MapKeyColumn;
import jakarta.persistence.Table;
import lombok.Data;

@Data
@Entity
@Table(name = "devices")
public class Device {

    @Id
    @Column(name = "device_eui", nullable = false, unique = true)
    private String deviceEUI;

    @Column(name = "lastUpdate")
    private Date lastUpdate;

    @Column(name = "currentTemperature")
    private Float currentTemperature;

    @ElementCollection
    @CollectionTable(
        name = "device_env_requests",
        joinColumns = @JoinColumn(name = "device_eui", referencedColumnName = "device_eui")
    )
    @MapKeyColumn(name = "device_env_key")
    @Column(name = "device_env_value")
    private Map<String, String> deviceEnvRequests = new HashMap<>();


    @JsonIgnore
    @ManyToOne
    @JoinColumn(name = "user", nullable = true)
    private User user;

    public Device(String EUI, User user) {
        this.deviceEUI = EUI;
        this.user = user;
        this.lastUpdate = new Date();
        this.currentTemperature = null;
        this.deviceEnvRequests.put("frequency", null);
        this.deviceEnvRequests.put("duty_frequency", null);
        this.deviceEnvRequests.put("temperature", null);
        this.deviceEnvRequests.put("start time", null);
        this.deviceEnvRequests.put("finish after", null);
    }
    
    public Map<String, String> getDeviceEnvRequests() {
        return deviceEnvRequests;
    }
    
    public void setDeviceEnvRequests(String key, String value) {
        if (deviceEnvRequests.containsKey(key)) {
            deviceEnvRequests.put(key, value);
        } else {
            throw new IllegalArgumentException("Chiave non valida: " + key);
        }
    }

}
