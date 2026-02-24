package com.anomaly.detection.backend.Repository;

import com.anomaly.detection.backend.Model.SystemSettings;
import org.springframework.data.mongodb.repository.MongoRepository;
import org.springframework.stereotype.Repository;

import java.util.Optional;

@Repository
public interface SystemSettingsRepository extends MongoRepository<SystemSettings,Long> {
    Optional<SystemSettings> findByConfigKey(String configKey);
}
