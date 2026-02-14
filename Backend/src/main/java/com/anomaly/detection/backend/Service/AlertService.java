package com.anomaly.detection.backend.Service;

import com.anomaly.detection.backend.Dto.AlertRequestDto;
import com.anomaly.detection.backend.Model.Alert;
import com.anomaly.detection.backend.Repository.AlertRepository;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

@Service
@Slf4j
public class AlertService {

    @Autowired
    private AlertRepository alertRepository;

    public Alert processAndSaveAlert(AlertRequestDto dto){
        try{
            Alert alert = new Alert();

            alert.setSourceIp(dto.getSourceIp());
            alert.setDestinationIp(dto.getDestinationIp());
            alert.setDetectedBy(dto.getModelName());
            alert.setAnomalyScore(dto.getMseScore());
            alert.setDescription(dto.getDescription() != null ? dto.getDescription() : "Manual anomaly identification");

            alert.setTimestamp(LocalDateTime.now());
            alert.setStatus("Open");
            alert.setSeverity(calculateSeverity(dto.getMseScore(), dto.getThreshold()));

            log.info("New intrusion alert registered from model: {} with severity: {}.", dto.getModelName(), alert.getSeverity());

            if("CRITICAL".equals(alert.getSeverity())){
                triggerActivePrevention(alert);
            }

            return alertRepository.save(alert);
        }catch(Exception e){
            log.error("Failed to save alert: {}", e.getMessage());
            throw new RuntimeException("Database error: Could not save alert.");
        }
    }

    /**
     * Calculates severity relative to the threshold.
     * Logic:
     * - CRITICAL: $score > threshold \times 3$
     * - HIGH: $score > threshold \times 1.5$
     * - MEDIUM: $score > threshold$
     */
    private String calculateSeverity(double score, double threshold){
        if(score > threshold*3){
            return "CRITICAL";  
        }
        if (score > threshold*1.5) {
            return "HIGH";
        }
        return "MEDIUM";
    }

    /**
     * Updates the status of an alert (Open, In Progress, Closed).
     */
    public Alert updateAlertStatus(String id, String newStatus){
        Alert alert = alertRepository.findById(id)
                .orElseThrow(() -> new RuntimeException("Alert not found with id: " + id));
        alert.setStatus(newStatus);
        log.info("Alert {} status updated to {}.", id, newStatus);
        return alertRepository.save(alert);
    }

    /**
     * Returns ALL alerts from the database, sorted by time (latest first).
     */
    public List<Alert> getAllAlerts(){
        List<Alert> alerts = alertRepository.findAll();
        if(alerts.isEmpty()){
            log.warn("No alerts found in the database.");
        }
        return alerts.stream()
                .sorted((a1,a2) -> a2.getTimestamp().compareTo(a1.getTimestamp()))
                .collect(Collectors.toList());
    }

    /**
     * Returns all alerts sorted by time (latest first) for the Dashboard feed[cite: 434].
     */
    public List<Alert> getLatestAlerts(int limit){
        List<Alert> alerts = alertRepository.findAll();

        if(alerts.isEmpty()){
            log.warn("No alerts found in the database.");
        }

        return alerts.stream()
                .sorted((a1,a2) -> a2.getTimestamp().compareTo(a1.getTimestamp()))
                .limit(limit)
                .collect(Collectors.toList());
    }

    /**
     * Aggregates alert counts by model for graphical display.
     */
    public Map<String, Long> getAlertStatsByModel(){
        Map<String,Long> stats = alertRepository.findAll().stream()
                .collect(Collectors.groupingBy(Alert::getDetectedBy, Collectors.counting()));

        if(stats.isEmpty()){
            log.info("Database is currently empty.");
        }
        return stats;
    }

    /**
     * Placeholder for future Firewall/IPS integration[cite: 354, 355].
     */
    private void triggerActivePrevention(Alert alert){
        log.warn("[PREVENTION] Critical anomaly from {}. Ready for firewall block logic.", alert.getSourceIp());
    }
}
