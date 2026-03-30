package com.anomaly.detection.backend.Service;


import com.anomaly.detection.backend.Model.SystemSettings;

public interface SystemSettingsService {
    SystemSettings getSettings();
    SystemSettings updateSettings(SystemSettings settings);
}
