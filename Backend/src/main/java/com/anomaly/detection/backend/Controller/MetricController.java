package com.anomaly.detection.backend.Controller;

import com.anomaly.detection.backend.Dto.MetricRequestDto;
import com.anomaly.detection.backend.Dto.MetricResponseDto;
import com.anomaly.detection.backend.Model.Metric;
import com.anomaly.detection.backend.Service.MetricService;
import com.anomaly.detection.backend.Service.MetricServiceImpl;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/metric")
@RequiredArgsConstructor
@CrossOrigin(origins = "*")
public class MetricController {

    private final MetricService metricService;

    @PostMapping("/report")
    public ResponseEntity<MetricResponseDto> reportMetric(@Valid @RequestBody MetricRequestDto metricDto) {
        return ResponseEntity.ok(metricService.saveMetric(metricDto));
    }

    @GetMapping
    public ResponseEntity<List<MetricResponseDto>> getHistory() {
        return ResponseEntity.ok(metricService.getAllMetrics());
    }

    @GetMapping("/model/{name}")
    public ResponseEntity<List<MetricResponseDto>> getByModel(@PathVariable String name) {
        return ResponseEntity.ok(metricService.getMetricsByModel(name));
    }
}
