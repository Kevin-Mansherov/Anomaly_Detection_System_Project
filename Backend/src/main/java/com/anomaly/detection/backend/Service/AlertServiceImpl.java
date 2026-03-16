package com.anomaly.detection.backend.Service;

import com.anomaly.detection.backend.Dto.AlertRequestDto;
import com.anomaly.detection.backend.Dto.AlertResponseDto;
import com.anomaly.detection.backend.Exception.ResourceNotFoundException;
import com.anomaly.detection.backend.Mapper.AlertMapper;
import com.anomaly.detection.backend.Model.Alert;
import com.anomaly.detection.backend.Repository.AlertRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.util.List;
import java.util.stream.Collectors;

@Service
@Slf4j
@RequiredArgsConstructor
public class AlertServiceImpl implements  AlertService{

    private final AlertRepository alertRepository;
    private final AlertMapper alertMapper;

    @Override
    public AlertResponseDto processAndSaveAlert(AlertRequestDto dto) {
        try {
            Alert alert = alertMapper.toEntity(dto);

            alert.setTimestamp(LocalDateTime.now());
            alert.setStatus("OPEN");
            alert.setSeverity(calculateSeverity(dto.getAnomalyScore()));

            if(alert.getDescription() == null || alert.getDescription().isEmpty()){
                alert.setDescription("Automatic anomaly detection alert.");
            }

            Alert savedAlert = alertRepository.save(alert);
            log.info("New alert generated: ID: {}, Severity: {}", savedAlert.getId(), savedAlert.getSeverity());

            return alertMapper.toResponseDto(savedAlert);
        }catch(Exception e){
            log.error("Failed to process alert from: {}: {}", dto.getDetectedBy(), e.getMessage());
            throw new RuntimeException("Alert processing failed.");
        }
    }

    @Override
    public List<AlertResponseDto> getAllAlerts(){
        return alertRepository.findAll().stream()
                .map(alertMapper::toResponseDto)
                .collect(Collectors.toList());
    }

    @Override
    public List<AlertResponseDto> getAlertByStatus(String status) {
        return alertRepository.findByStatus(status).stream()
                .map(alertMapper::toResponseDto)
                .collect(Collectors.toList());
    }

    @Override
    public void updateAlertStatus(String id, String status) {
        Alert alert = alertRepository.findById(id)
                .orElseThrow(() -> new ResourceNotFoundException("Alert not found with id: " + id));

        alert.setStatus(status);
        alertRepository.save(alert);
        log.info("Alert {} status updated to {}", id,status);
    }

    @Override
    public String calculateSeverity(double score) {
        if(score > 10.0){
            return "CRITICAL";
        }
        if(score > 5.0){
            return "HIGH";
        }
        return "MEDIUM";
    }
}
