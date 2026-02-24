package com.anomaly.detection.backend.Controller;

import com.anomaly.detection.backend.Model.Metric;
import com.anomaly.detection.backend.Service.MetricService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/metric")
@CrossOrigin(origins = "*")
public class MetricController {
    @Autowired
    private MetricService metricService;

    @PostMapping("/report")
    public Metric reportMetric(@RequestParam Metric metric){
        return metricService.saveMetric(metric);
    }

    @GetMapping
    public List<Metric> getHistory(){
        return metricService.getAllMetrics();
    }
}
