package com.anomaly.detection.backend.Service;

import com.anomaly.detection.backend.Model.SystemSettings;
import com.anomaly.detection.backend.Repository.SystemSettingsRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

@Service
public class SystemSettingsService {
    @Autowired
    private SystemSettingsRepository systemSettingsRepository;

    public double getThreshold(String modelName) {
        return systemSettingsRepository.findByConfigKey(modelName + "_threshold")
                .map(setting -> Double.parseDouble(setting.getConfigValue()))
                .orElse(0.05); // Default threshold
    }

    public SystemSettings updateSettings(String key, String value){
        SystemSettings setting = systemSettingsRepository.findByConfigKey(key)
                .orElse(new SystemSettings());
        setting.setConfigKey(key);
        setting.setConfigValue(value);
        return systemSettingsRepository.save(setting);
    }
}
