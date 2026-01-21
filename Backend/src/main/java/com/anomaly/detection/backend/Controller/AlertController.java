package com.anomaly.detection.backend.Controller;

import com.anomaly.detection.backend.Dto.AlertRequestDto;
import com.anomaly.detection.backend.Model.Alert;
import com.anomaly.detection.backend.Service.AlertService;
import jakarta.validation.Valid;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/alerts")
@CrossOrigin(origins = "*")
public class AlertController
{
    @Autowired
    private AlertService alertService;

    @PostMapping
    public ResponseEntity<Alert> createAlert(@Valid @RequestBody AlertRequestDto alertDto){
        Alert savedAlert = alertService.proccessAndSaveAlert(alertDto);
        return new ResponseEntity<>(savedAlert, HttpStatus.CREATED);
    }
}
