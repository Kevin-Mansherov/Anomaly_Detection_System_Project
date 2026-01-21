package com.anomaly.detection.backend.Service;

import com.anomaly.detection.backend.Dto.AlertRequestDto;
import com.anomaly.detection.backend.Model.Alert;
import com.anomaly.detection.backend.Repository.AlertRepository;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.util.List;

@Service
@Slf4j
public class AlertService {

    @Autowired
    private AlertRepository alertRepository;

    public Alert proccessAndSaveAlert(AlertRequestDto dto){
        try{
            Alert alert = new Alert();

            alert.setSourceIp(dto.getSourceIp());
            alert.setDestinationIp(dto.getDestinationIp());
            alert.setDetectedBy(dto.getDetectedBy());
            alert.setAnomalyScore(dto.getAnomalyScore());
            alert.setDescription(dto.getDescription() != null ? dto.getDescription() : "No description provided");

            alert.setTimestamp(LocalDateTime.now());
            alert.setStatus("Open");
            alert.setSeverity(calculateSeverity(dto.getAnomalyScore()));

            log.info("Saving new alert from model: {}", dto.getDetectedBy());
            return alertRepository.save(alert);
        }catch(Exception e){
            log.error("Failed to save alert: {}", e.getMessage());
            throw new RuntimeException("Database error: Could not save alert.");
        }
    }

    private String calculateSeverity(double score){
        if(score > 10.0){
            return "CRITICAL";
        }
        if(score > 5.0){
            return "HIGH";
        }
        return "MEDIUM";
    }

    private List<Alert> getAllAlerts(){
        return alertRepository.findAll();
    }
}
