package com.anomaly.detection.backend.Dto;

import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
public class AlertRequestDto {

    @NotBlank(message = "Source IP is required")
    private String sourceIp;

    @NotBlank(message = "Destination IP is required")
    private String destinationIp;

    @NotBlank(message = "Model name is required")
    private String detectedBy;

    @NotNull(message = "Anomaly Score cannot be null")
    @Min(value = 0, message = "Anomaly Score must be non-negative")
    private double anomalyScore;

    private String description;
}
