package com.anomaly.detection.backend.Service;

import com.anomaly.detection.backend.Dto.MetricRequestDto;
import com.anomaly.detection.backend.Dto.MetricResponseDto;

import java.util.List;

public interface MetricService {
    MetricResponseDto saveMetric(MetricRequestDto dto);
    List<MetricResponseDto> getAllMetrics();
    List<MetricResponseDto> getMetricsByModel(String modelName);
}
