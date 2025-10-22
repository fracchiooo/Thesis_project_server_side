package com.server.backend.dto;

import java.util.Date;

import com.fasterxml.jackson.annotation.JsonFormat;

import io.micrometer.common.lang.NonNull;
import jakarta.annotation.Nonnull;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class MessageUplinkDto {

    @JsonFormat(pattern = "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'", timezone = "UTC")
    private Date lastUpdate;

    @Nonnull
    private Float currentTemperature;

    @Nonnull
    private Float currentSensedFrequency;

    @Nonnull
    private DeviceEnvMessageDto  deviceEnvRequests;
    
}
