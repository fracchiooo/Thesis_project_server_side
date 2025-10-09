package com.server.backend.dto;


import jakarta.annotation.Nonnull;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class userProfileRequestDto {

    @Nonnull
    private String password;
    
    @Nonnull
    private String username;
    
}
