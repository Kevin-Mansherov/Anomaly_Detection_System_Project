package com.anomaly.detection.backend.Repository;


import com.anomaly.detection.backend.Model.Metric;
import org.springframework.data.mongodb.repository.MongoRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface MetricRepository extends MongoRepository<Metric, String> {
    List<Metric> findTop10ByOrderByTimestampDesc();
}
