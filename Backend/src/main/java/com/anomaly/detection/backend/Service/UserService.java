package com.anomaly.detection.backend.Service;

import com.anomaly.detection.backend.Dto.AuthResponseDto;
import com.anomaly.detection.backend.Dto.UserRequestDto;
import com.anomaly.detection.backend.Dto.UserResponseDto;
import com.anomaly.detection.backend.Model.User;

import java.util.List;

public interface UserService {
    AuthResponseDto registerUser(UserRequestDto dto);
    List<UserResponseDto> getAllUsers();
    UserResponseDto getUserById(String id);
    UserResponseDto updateUser(String id, UserRequestDto userDto);
    void deleteUser(String id);
}
