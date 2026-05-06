package com.anomaly.detection.backend.Service;

import com.anomaly.detection.backend.Model.SystemSettings;
import com.anomaly.detection.backend.Repository.SystemSettingsRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

@Service
@RequiredArgsConstructor
public class SystemSettingsServiceImpl implements  SystemSettingsService{

    private final SystemSettingsRepository repository;

    @Override
    public SystemSettings getSettings() {
        return repository.findAll().stream().findFirst()
                .orElseGet(() -> {
                    SystemSettings defaultSettings = new SystemSettings();
                    defaultSettings.setPacketThreshold(3.369822);
                    defaultSettings.setFlowThreshold(14.386914);
                    defaultSettings.setDetectionEnabled(true);
                    return repository.save(defaultSettings);
                });
    }

    @Override
    public SystemSettings updateSettings(SystemSettings settings) {
        SystemSettings existing = getSettings();

        existing.setPacketThreshold(settings.getPacketThreshold());
        existing.setFlowThreshold(settings.getFlowThreshold());

        existing.setDetectionEnabled(settings.isDetectionEnabled());
        existing.setLogRetentionDays(settings.getLogRetentionDays());
        return repository.save(existing);
    }
}
