package com.server.backend.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class modelResponseDto {

    String status;

    Object received;

    Object result;
    
}
