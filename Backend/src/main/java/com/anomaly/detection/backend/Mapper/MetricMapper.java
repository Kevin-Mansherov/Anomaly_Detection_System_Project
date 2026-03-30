package com.anomaly.detection.backend.Mapper;

import com.anomaly.detection.backend.Dto.MetricRequestDto;
import com.anomaly.detection.backend.Dto.MetricResponseDto;
import com.anomaly.detection.backend.Model.Metric;
import org.springframework.stereotype.Component;

@Component
public class MetricMapper {

    public Metric toEntity(MetricRequestDto dto) {
        if (dto == null) return null;
        Metric metric = new Metric();
        metric.setModelName(dto.getModelName());
        metric.setMseValue(dto.getMseValue());
        metric.setThresholdLimit(dto.getThresholdLimit());
        return metric;
    }

    public MetricResponseDto toResponseDto(Metric metric) {
        if (metric == null) return null;
        MetricResponseDto dto = new MetricResponseDto();
        dto.setId(metric.getId());
        dto.setTimestamp(metric.getTimestamp());
        dto.setModelName(metric.getModelName());
        dto.setMseValue(metric.getMseValue());
        dto.setThresholdLimit(metric.getThresholdLimit());
        dto.setAnomaly(metric.isAnomaly());
        return dto;
    }
}
