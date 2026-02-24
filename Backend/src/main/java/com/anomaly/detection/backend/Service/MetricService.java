package com.anomaly.detection.backend.Service;


import com.anomaly.detection.backend.Model.Metric;
import com.anomaly.detection.backend.Repository.MetricRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class MetricService {

    @Autowired
    private MetricRepository metricRepository;

    public Metric saveMetric(Metric metric) {
        metric.setAnomaly(metric.getMseValue() > metric.getThresholdLimit());
        return metricRepository.save(metric);
    }

    public List<Metric> getAllMetrics(){
        return metricRepository.findAll();
    }
}
