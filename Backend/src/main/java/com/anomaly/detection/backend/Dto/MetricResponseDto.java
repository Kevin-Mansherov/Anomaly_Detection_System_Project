package com.anomaly.detection.backend.Dto;

import lombok.Data;

import java.time.LocalDateTime;

@Data
public class MetricResponseDto {
    private String id;
    private LocalDateTime timestamp;
    private String modelName;
    private double mseValue;
    private double thresholdLimit;
    private boolean isAnomaly;
}
