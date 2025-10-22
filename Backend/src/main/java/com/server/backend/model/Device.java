package com.server.backend.model;

import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.HashMap;
import java.util.Map;
import java.util.TimeZone;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.JoinColumn;
import jakarta.persistence.ManyToOne;
import jakarta.persistence.PostLoad;
import jakarta.persistence.PrePersist;
import jakarta.persistence.PreUpdate;
import jakarta.persistence.Table;
import jakarta.persistence.Transient;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@NoArgsConstructor
@AllArgsConstructor
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

    @Column(name = "currentSensedfrequency")
    private Float currentSensedFrequency;


    @Column(name = "device_env_requests", columnDefinition = "TEXT")
    private String deviceEnvRequestsJson;

    @Transient
    private Map<String, Object> deviceEnvRequests = new HashMap<>();

    @Transient
    private static final ObjectMapper objectMapper = new ObjectMapper();


    @JsonIgnore
    @ManyToOne
    @JoinColumn(name = "user_id", referencedColumnName = "username")
    private User user;

    public Device(String EUI, User user) {
        this.deviceEUI = EUI;
        this.user = user;
        this.lastUpdate = new Date();
        this.currentTemperature = null;
        this.currentSensedFrequency = null;
        this.deviceEnvRequests.put("frequency", null);
        this.deviceEnvRequests.put("duty_frequency", null);
        this.deviceEnvRequests.put("start time", null);
        this.deviceEnvRequests.put("finish after", null);
    }
    
    public Map<String, Object> getDeviceEnvRequests() {
        if (deviceEnvRequests == null && deviceEnvRequestsJson != null) {
            try {
                deviceEnvRequests = objectMapper.readValue(deviceEnvRequestsJson, new TypeReference<Map<String, Object>>() {});
            } catch (JsonProcessingException e) {
                deviceEnvRequests = new HashMap<>();
            }
        }
        return deviceEnvRequests;
    }
    
    public void setDeviceEnvRequests(String key, Object value) {
        if (deviceEnvRequests == null) {
            deviceEnvRequests = new HashMap<>();
        }
        switch (key) {
            case "frequency", "duty_frequency", "finish after" -> {
                if (value instanceof Number number) {
                    deviceEnvRequests.put(key, number.floatValue());
                } else {
                    throw new IllegalArgumentException("Valore non valido per " + key + ": deve essere un numero decimale.");
                }
            }
            case "start time" -> {
                if (value instanceof Date date) {
                    SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSS'Z'");
                    sdf.setTimeZone(TimeZone.getTimeZone("UTC"));
                    deviceEnvRequests.put(key, sdf.format(date));
                } else {
                    throw new IllegalArgumentException("Valore non valido per " + key + ": deve essere una data.");
                }
            }
            default -> throw new IllegalArgumentException("Chiave non valida: " + key);
        }
        try {
            this.deviceEnvRequestsJson = objectMapper.writeValueAsString(deviceEnvRequests);
            this.setDeviceEnvRequestsJson(this.deviceEnvRequestsJson);
        } catch (JsonProcessingException e) {
            throw new RuntimeException("Errore nella serializzazione JSON", e);
        }
    }

    @SuppressWarnings("unused")
    @PrePersist
    @PreUpdate
    private void serializeMap() {
        if (deviceEnvRequests != null) {
            try {
                this.deviceEnvRequestsJson = objectMapper.writeValueAsString(deviceEnvRequests);
            } catch (JsonProcessingException e) {
                throw new RuntimeException("Errore nella serializzazione JSON", e);
            }
        }
    }

    @SuppressWarnings("unused")
    @PostLoad
    private void deserializeMap() {
        if (deviceEnvRequestsJson != null) {
            try {
                this.deviceEnvRequests = objectMapper.readValue(deviceEnvRequestsJson, new TypeReference<Map<String, Object>>() {});
            } catch (JsonProcessingException e) {
                this.deviceEnvRequests = new HashMap<>();
            }
        }
    }

}
