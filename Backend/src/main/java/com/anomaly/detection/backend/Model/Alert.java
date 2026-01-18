package com.anomaly.detection.backend.Model;

import jakarta.persistence.Id;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.springframework.data.mongodb.core.mapping.Document;

import java.time.LocalDateTime;

@Data
@AllArgsConstructor
@NoArgsConstructor
@Document(collation = "alerts")
public class Alert {
    @Id
    private String id;
    private LocalDateTime timestamp;
    private String sourceIp;
    private String destinationIp;
    private String detectedBy;
    private double anomalyScore;
    private String status;
    private String description;
}
