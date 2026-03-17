package com.anomaly.detection.backend.Service;


import com.anomaly.detection.backend.Dto.MetricRequestDto;
import com.anomaly.detection.backend.Dto.MetricResponseDto;
import com.anomaly.detection.backend.Mapper.MetricMapper;
import com.anomaly.detection.backend.Model.Metric;
import com.anomaly.detection.backend.Repository.MetricRepository;
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
public class MetricServiceImpl implements MetricService {

    private final MetricRepository metricRepository;
    private final MetricMapper metricMapper;


    @Override
    public MetricResponseDto saveMetric(MetricRequestDto dto) {
        Metric metric = metricMapper.toEntity(dto);

        metric.setTimestamp(LocalDateTime.now());
        metric.setAnomaly(metric.getMseValue() > metric.getThresholdLimit());

        Metric savedMetric = metricRepository.save(metric);
        log.info("Metric saved for model: {}. Anomaly detected: {}", savedMetric.getModelName(), savedMetric.isAnomaly());

        return metricMapper.toResponseDto(savedMetric);
    }

    @Override
    public List<MetricResponseDto> getAllMetrics() {
        return metricRepository.findAll().stream()
                .map(metricMapper::toResponseDto)
                .collect(Collectors.toList());
    }

    @Override
    public List<MetricResponseDto> getMetricsByModel(String modelName) {
        return metricRepository.findByModelName(modelName).stream()
                .map(metricMapper::toResponseDto)
                .collect(Collectors.toList());
    }
}
