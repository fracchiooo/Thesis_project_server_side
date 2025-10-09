package com.server.backend.dto;

import java.util.List;
import com.server.backend.model.Device;
import com.server.backend.model.Prediction;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class userProfileResponseDto {

    private String username;
    private List<Device> devices;
    private List<Prediction> predictions;
    
}
