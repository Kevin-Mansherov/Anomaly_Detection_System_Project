package com.anomaly.detection.backend.Dto;

import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
public class AlertRequestDto {

    @NotBlank
    private String modelName;

    @NotBlank
    private String severity;

    @NotNull
    private Double mseScore;

    @NotNull
    private Double threshold;

    @NotBlank(message = "Source IP is required")
    private String sourceIp;

    @NotBlank(message = "Destination IP is required")
    private String destinationIp;

    private String description;
}
