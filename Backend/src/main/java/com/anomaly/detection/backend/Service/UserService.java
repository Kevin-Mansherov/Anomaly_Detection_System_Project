package com.anomaly.detection.backend.Service;

import com.anomaly.detection.backend.Dto.UserResponseDto;
import com.anomaly.detection.backend.Model.User;

import java.util.List;

public interface UserService {
    UserResponseDto registerUser(User user);
    List<UserResponseDto> getAllUsers();
    UserResponseDto getUserById(String id);
    UserResponseDto updateUser(String id, User userDetails);
    void deleteUser(String id);
}
