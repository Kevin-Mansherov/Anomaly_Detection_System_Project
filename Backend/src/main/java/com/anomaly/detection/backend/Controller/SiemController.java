package com.anomaly.detection.backend.Controller;

import com.anomaly.detection.backend.Dto.AlertRequestDto;
import com.anomaly.detection.backend.Dto.LogEventDto;
import com.anomaly.detection.backend.Dto.PythonAnalysisResponse;
import com.anomaly.detection.backend.Model.Alert;
import com.anomaly.detection.backend.Repository.AlertRepository;
import com.anomaly.detection.backend.Service.AlertService;
import com.anomaly.detection.backend.Service.UbaAnalysisService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.time.LocalDateTime;

@RestController
@RequestMapping("/api/logs")
public class SiemController {

    @Autowired
    private UbaAnalysisService ubaService;

    @Autowired
    private AlertService alertService;

    @PostMapping
    public ResponseEntity<String> receiveLogEvent(@RequestBody LogEventDto logEvent){
        System.out.println("[SIEM] Received new log event: " + logEvent.getEvent_id());

        PythonAnalysisResponse analysis = ubaService.analyzeLogWithPython(logEvent);

        if(analysis!=null && analysis.is_anomaly()){
            System.out.println("[SIEM] Policy Anomaly Detected!!!");

            AlertRequestDto newAlertDto = new AlertRequestDto();
            newAlertDto.setSourceIp(logEvent.getPc());
            newAlertDto.setDestinationIp("N/A");
            newAlertDto.setDetectedBy("Policy Engine (UBA)");
            newAlertDto.setAnomalyScore(analysis.getEnergy_score());
            newAlertDto.setDescription("User " + logEvent.getUser() + " performed anomalous " + logEvent.getActivity() + " activity.");

            alertService.processAndSaveAlert(newAlertDto);
        }

        return ResponseEntity.ok("Log processed");
    }

    @PostMapping("/alert")
    public ResponseEntity<String> receiveNetworkAlert(@RequestBody AlertRequestDto dto){
        System.out.println("[SIEM] Received DIRECT NETWORK ALERT from: " + dto.getDetectedBy());

        alertService.processAndSaveAlert(dto);

        return ResponseEntity.ok("Network Alert received and saved");
    }
}
