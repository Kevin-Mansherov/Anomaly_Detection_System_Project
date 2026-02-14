package com.anomaly.detection.backend.Controller;

import com.anomaly.detection.backend.Dto.AlertRequestDto;
import com.anomaly.detection.backend.Model.Alert;
import com.anomaly.detection.backend.Service.AlertService;
import jakarta.validation.Valid;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/alerts")
@CrossOrigin(origins = "*")
public class AlertController
{
    @Autowired
    private AlertService alertService;

    //Receive alert from Python Inference Engine
    @PostMapping
    public ResponseEntity<Alert> createAlert(@Valid @RequestBody AlertRequestDto alertDto){
        Alert savedAlert = alertService.processAndSaveAlert(alertDto);

        return new ResponseEntity<>(savedAlert, HttpStatus.CREATED);
    }

    //Fetch all alerts for the React Table
    @GetMapping("/all")
    public ResponseEntity<?> getAllAlerts(){
        List<Alert> alerts = alertService.getAllAlerts();
        if(alerts.isEmpty()){
            return new ResponseEntity<>("No alerts found.", HttpStatus.OK);
        }
        return new ResponseEntity<>(alerts, HttpStatus.OK);
    }

    @GetMapping("/latest")
    public ResponseEntity<?> getLatestAlerts(@RequestParam(defaultValue = "10") int limit) {
        List<Alert> alerts = alertService.getLatestAlerts(limit);
        if(alerts.isEmpty()){
            return new ResponseEntity<>("No alerts found.", HttpStatus.OK);
        }
        return new ResponseEntity<>(alerts, HttpStatus.OK);    }

    //Update alert status (Open/In Progress/Closed)
    @PutMapping("/{id}/status")
    public ResponseEntity<?> updateAlertStatus(@PathVariable String id, @RequestParam String status){
        try{
            Alert updated = alertService.updateAlertStatus(id, status);
            return new ResponseEntity<>(updated, HttpStatus.OK);
        }catch(RuntimeException e){
            return ResponseEntity.status(HttpStatus.NOT_FOUND).body(e.getMessage());
        }
    }

    //Get stats for Dashboard Charts
    @GetMapping("/stats")
    public ResponseEntity<?> getStats(){
        Map<String, Long> stats = alertService.getAlertStatsByModel();
        if(stats.isEmpty()){
            return ResponseEntity.status(HttpStatus.OK).body("No statistics available, database is empty.");
        }
        return new ResponseEntity<>(stats, HttpStatus.OK);
    }
}
