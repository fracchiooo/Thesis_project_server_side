package com.server.backend.dto;

import java.util.Date;

import jakarta.annotation.Nonnull;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class CommandDto {

    @Nonnull
    private Float frequency;

    @Nonnull
    private Float duty_frequency;

    @Nonnull
    private Float finish_after;
    
    private Date startTime;    
}
