package com.anomaly.detection.backend.Controller;

import com.anomaly.detection.backend.Model.SystemSettings;
import com.anomaly.detection.backend.Service.SystemSettingsService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/settings")
@RequiredArgsConstructor
public class SystemSettingsController {

    private final SystemSettingsService settingsService;

    @GetMapping
    public ResponseEntity<SystemSettings> getSettings(){
        return ResponseEntity.ok(settingsService.getSettings());
    }

    @PutMapping
    public ResponseEntity<SystemSettings> updateSettings(@RequestBody SystemSettings settings){
        return ResponseEntity.ok(settingsService.updateSettings(settings));
    }
}
