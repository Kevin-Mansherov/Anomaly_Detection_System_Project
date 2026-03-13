package com.anomaly.detection.backend.Model;


import lombok.Data;
import org.springframework.data.annotation.Id;
import org.springframework.data.mongodb.core.mapping.Document;

@Data
@Document(collection = "system_settings")
public class SystemSettings {
    @Id
    private Long id;

    private double globalThreshold;
    private int logRetentionDays;
    private boolean isDetectionEnabled;
    private String systemVersion;
}
