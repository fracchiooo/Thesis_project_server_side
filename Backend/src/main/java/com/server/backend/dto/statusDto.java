package com.server.backend.dto;

import java.util.Date;
import java.util.Map;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class statusDto {

    private String deviceEUI;

    private Date lastUpdate;

    private Float currentTemperature;

    private String username;

    private Map<String, String> deviceEnvRequests;

}
