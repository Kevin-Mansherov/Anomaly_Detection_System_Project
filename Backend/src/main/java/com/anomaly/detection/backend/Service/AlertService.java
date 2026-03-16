package com.anomaly.detection.backend.Service;

import com.anomaly.detection.backend.Dto.AlertRequestDto;
import com.anomaly.detection.backend.Dto.AlertResponseDto;

import java.util.List;

public interface AlertService {
    AlertResponseDto processAndSaveAlert(AlertRequestDto dto);
    List<AlertResponseDto> getAllAlerts();
    List<AlertResponseDto> getAlertByStatus(String status);
    void updateAlertStatus(String id, String status);
    String calculateSeverity(double score);
}
