package com.anomaly.detection.backend.Dto;


import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.Setter;

@AllArgsConstructor
@Getter
@Setter
public class PythonAnalysisResponse {
    private String status;
    private String event_id;
    private boolean is_anomaly;
    private double energy_score;

}
