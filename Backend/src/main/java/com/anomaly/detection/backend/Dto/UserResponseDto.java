package com.anomaly.detection.backend.Dto;

import lombok.Data;

import java.time.LocalDateTime;

@Data
public class UserResponseDto {
    private String id;
    private String username;
    private String role;
    private LocalDateTime lastLogin;
}
