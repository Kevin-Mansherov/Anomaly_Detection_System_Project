package com.anomaly.detection.backend.Dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import lombok.Data;

@Data
public class MetricRequestDto {
    @NotBlank(message = "Model name is required")
    private String modelName;

    @NotNull(message = "MSE value is required")
    private double mseValue;

    @NotNull(message = "Threshold limit is required")
    private double thresholdLimit;
}
