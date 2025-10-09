package com.server.backend.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class commandDto {

    private Float frequency;
    private Float duty_frequency;
    private Float temperature;
    private Float finish_after;
    
}
