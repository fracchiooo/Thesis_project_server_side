package com.server.backend.dto;

import java.util.Date;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class PredictResponseDto {

    private Long id;

    private Date timestamp;

    private Float predictedConcentration;

    private Float predictedUncertainty;

    private Float initialConcentration;

    private Float frequency;
    
    private Float dutyCycle;

    private Float timeLasted;

    private Float temperature;
}
