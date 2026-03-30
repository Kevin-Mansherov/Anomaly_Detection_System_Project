package com.anomaly.detection.backend.Model;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.springframework.data.annotation.Id;
import org.springframework.data.mongodb.core.mapping.Document;

import java.time.LocalDateTime;

@Data
@AllArgsConstructor
@NoArgsConstructor
@Document(collection = "metrics")
public class Metric {
    @Id
    private String id;
    private LocalDateTime timestamp;
    private String modelName;
    private double mseValue;
    private double thresholdLimit;
    private boolean isAnomaly;
}
