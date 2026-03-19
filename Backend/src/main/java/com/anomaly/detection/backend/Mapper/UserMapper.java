package com.anomaly.detection.backend.Mapper;

import com.anomaly.detection.backend.Dto.UserRequestDto;
import com.anomaly.detection.backend.Dto.UserResponseDto;
import com.anomaly.detection.backend.Model.User;
import org.springframework.stereotype.Component;

@Component
public class UserMapper {

    public UserResponseDto toResponseDto(User user){
        if(user == null){
            return null;
        }

        UserResponseDto dto = new UserResponseDto();
        dto.setId(user.getId());
        dto.setUsername(user.getUsername());
        dto.setRole(user.getRole());
        dto.setLastLogin(user.getLastLogin());

        return dto;
    }

//    public User toEntity(UserResponseDto dto){
//        if(dto == null){
//            return null;
//        }
//
//        User user = new User();
//        user.setId(dto.getId());
//        user.setUsername(dto.getUsername());
//        user.setRole(dto.getRole());
//        user.setLastLogin(dto.getLastLogin());
//
//        return user;
//    }

    public UserResponseDto toUserResponseDto(User user) {
        if (user == null) {
            return null;
        }
        UserResponseDto dto = new UserResponseDto();
        dto.setId(user.getId());
        dto.setUsername(user.getUsername());
        dto.setRole(user.getRole());
        return dto;
    }

    public User toEntity(UserRequestDto dto) {
        if (dto == null) return null;

        User user = new User();
        user.setUsername(dto.getUsername());
        user.setPassword(dto.getPassword()); // הסיסמה תוצפן בתוך ה-Service
        user.setRole(dto.getRole() != null ? dto.getRole() : "ROLE_USER");
        return user;
    }
}
