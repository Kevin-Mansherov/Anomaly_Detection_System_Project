package com.anomaly.detection.backend.Mapper;

import com.anomaly.detection.backend.Dto.AlertRequestDto;
import com.anomaly.detection.backend.Dto.AlertResponseDto;
import com.anomaly.detection.backend.Model.Alert;
import org.springframework.stereotype.Component;

@Component
public class AlertMapper {

    public Alert toEntity(AlertRequestDto dto){
        if(dto == null){
            return null;
        }

        Alert alert = new Alert();
        alert.setSourceIp(dto.getSourceIp());
        alert.setDestinationIp(dto.getDestinationIp());
        alert.setDetectedBy(dto.getDetectedBy());
        alert.setAnomalyScore(dto.getAnomalyScore());
        alert.setDescription(dto.getDescription());

        return alert;
    }

    public AlertResponseDto toResponseDto(Alert alert){
        if(alert == null){
            return null;
        }

        AlertResponseDto dto = new AlertResponseDto();
        dto.setId(alert.getId());
        dto.setTimestamp(alert.getTimestamp());
        dto.setSourceIp(alert.getSourceIp());
        dto.setDestinationIp(alert.getDestinationIp());
        dto.setDetectedBy(alert.getDetectedBy());
        dto.setAnomalyScore(alert.getAnomalyScore());
        dto.setStatus(alert.getStatus());
        dto.setDescription(alert.getDescription());
        dto.setSeverity(alert.getSeverity());

        return dto;
    }
}
