package com.anomaly.detection.backend.Model;

import org.springframework.data.annotation.Id;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.springframework.data.mongodb.core.mapping.Document;

import java.time.LocalDateTime;

@Data
@AllArgsConstructor
@NoArgsConstructor
@Document(collection = "alerts")
public class Alert {
    @Id
    private String id;
    private LocalDateTime timestamp;
    private String sourceIp; //From scapy
    private String destinationIp; //From scapy
    private String detectedBy; //Model name or detection method
    private double anomalyScore; //Maps to MSE Score
    private double threshold; //Safety limit
    private String status;
    private String description;
    private String severity; //"CRITICAL" or "WARNING"
}
