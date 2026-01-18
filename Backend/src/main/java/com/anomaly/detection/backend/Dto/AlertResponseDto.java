package com.anomaly.detection.backend.Dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class AlertResponseDto {
    private String sourceIp;
    private String destinationIp;
    private String detectedBy;
    private double anomalyScore;
    private String description;
}
