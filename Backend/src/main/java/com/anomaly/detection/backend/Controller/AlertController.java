package com.anomaly.detection.backend.Controller;

import com.anomaly.detection.backend.Dto.AlertRequestDto;
import com.anomaly.detection.backend.Dto.AlertResponseDto;
import com.anomaly.detection.backend.Model.Alert;
import com.anomaly.detection.backend.Service.AlertService;
import com.anomaly.detection.backend.Service.AlertServiceImpl;
import jakarta.validation.Valid;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/alerts")
@CrossOrigin(origins = "*")
public class AlertController
{
    @Autowired
    private AlertService alertService;

    @PostMapping
    public ResponseEntity<AlertResponseDto> createAlert(@Valid @RequestBody AlertRequestDto alertDto){
        AlertResponseDto savedAlert = alertService.processAndSaveAlert(alertDto);
        return new ResponseEntity<>(savedAlert, HttpStatus.CREATED);
    }

    @GetMapping
    public ResponseEntity<List<AlertResponseDto>> getAllAlerts(){
        List<AlertResponseDto> alerts = alertService.getAllAlerts();
        return new ResponseEntity<>(alerts, HttpStatus.OK);
    }

    @GetMapping("/status/{status}")
    public ResponseEntity<List<AlertResponseDto>> getByStatus(@PathVariable String status){
        return ResponseEntity.ok(alertService.getAlertByStatus(status));
    }

    @PatchMapping("/{id}/status")
    public ResponseEntity<Void> updateStatus(@PathVariable String id, @RequestParam String status){
        alertService.updateAlertStatus(id, status);
        return ResponseEntity.noContent().build();
    }
}
