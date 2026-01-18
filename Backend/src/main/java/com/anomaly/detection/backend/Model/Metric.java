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
@Document(collection = "metrics")
public class Metric {
    @Id
    private String id;
    private LocalDateTime timestamp;
    private double avgError; //Average reconstruction error for time-series graphs
    private int trafficVolume; //Total monitored traffic volume
    private int activeConnections; //Number of active connections at sample time
}
