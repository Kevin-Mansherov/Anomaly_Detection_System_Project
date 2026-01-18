package com.anomaly.detection.backend.Repository;

import com.anomaly.detection.backend.Model.Alert;
import org.springframework.data.mongodb.repository.MongoRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface AlertRepository extends MongoRepository<Alert,String> {
    List<Alert> findByStatus(String status);
    List<Alert> findByDetectedBy(String detectedBy);
}
