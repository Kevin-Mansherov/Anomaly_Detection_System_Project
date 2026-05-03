package com.anomaly.detection.backend.Dto;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.Setter;

@AllArgsConstructor
@Getter
@Setter
public class LogEventDto {
    private String event_id;
    private String user;
    private String timestamp;
    private String activity;
    private String pc;
    private String source;
}
