package com.anomaly.detection.backend.Dto;

import lombok.Data;

import java.time.LocalDateTime;

@Data
public class AlertResponseDto {
    private String id;
    private LocalDateTime timestamp;
    private String sourceIp;
    private String destinationIp;
    private String detectedBy;
    private double anomalyScore;
    private String status;
    private String description;
    private String severity;
}
