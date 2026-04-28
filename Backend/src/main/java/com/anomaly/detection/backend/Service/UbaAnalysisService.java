package com.anomaly.detection.backend.Service;

import com.anomaly.detection.backend.Dto.LogEventDto;
import com.anomaly.detection.backend.Dto.PythonAnalysisResponse;
import lombok.AllArgsConstructor;
import lombok.NoArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

@Service
@AllArgsConstructor
public class UbaAnalysisService {
    private final RestTemplate restTemplate;

    private final String PYTHON_API_URL = "http://localhost:8000/api/analyze_log";

    public UbaAnalysisService(){
        this.restTemplate = new RestTemplate();
    }

    public PythonAnalysisResponse analyzeLogWithPython(LogEventDto logEvent){
        try{
            ResponseEntity<PythonAnalysisResponse> response = restTemplate.postForEntity(
                    PYTHON_API_URL,
                    logEvent,
                    PythonAnalysisResponse.class
            );
            return response.getBody();
        }catch(Exception e){
            System.err.println("[ERROR] Failed to communicate with Python Microservice: " + e.getMessage());
            return null;
        }
    }
}
