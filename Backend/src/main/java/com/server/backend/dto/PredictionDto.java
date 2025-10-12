package com.server.backend.dto;

import jakarta.annotation.Nonnull;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class PredictionDto {

    @Nonnull
    private Float initialConcentration;

    @Nonnull
    private Float frequency;
    
    @Nonnull
    private Float dutyCycle;

    @Nonnull
    private Float timeLasted;

    @Nonnull
    private Float temperature;
    
}
