package com.anomaly.detection.backend.Service;


import com.anomaly.detection.backend.Dto.UserResponseDto;
import com.anomaly.detection.backend.Exception.ResourceNotFoundException;
import com.anomaly.detection.backend.Mapper.UserMapper;
import com.anomaly.detection.backend.Model.User;
import com.anomaly.detection.backend.Repository.UserRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

@Service
@Slf4j
@RequiredArgsConstructor
public class UserServiceImpl implements  UserService {

    private final UserRepository userRepository;
    private final UserMapper userMapper;

    @Override
    public UserResponseDto registerUser(User user){
        if(userRepository.existsByUsername(user.getUsername())){
            log.info("Registration failed: Username {} already exists", user.getUsername());
            throw new RuntimeException("Username already exists");
        }

        user.setCreatedAt(LocalDateTime.now());
        user.setUpdatedAt(LocalDateTime.now());

        User savedUser = userRepository.save(user);
        log.info("New user registered: {}", user.getUsername());

        return userMapper.toResponseDto(savedUser);
    }

    @Override
    public List<UserResponseDto> getAllUsers(){
        return userRepository.findAll().stream()
                .map(userMapper::toResponseDto)
                .collect(Collectors.toList());
    }

    @Override
    public UserResponseDto getUserById(String id){
        Optional<User> user = userRepository.findById(id);
        if(user.isEmpty()){
            throw new RuntimeException("User not found with id: " + id);
        }
        return userMapper.toResponseDto(user.get());
    }

    @Override
    public UserResponseDto updateUser(String id, User userDetails){
        User user = userRepository.findById(id)
                .orElseThrow(() -> new ResourceNotFoundException("User not found with id: " + id));

        user.setUsername(userDetails.getUsername());
        user.setRole(userDetails.getRole());
        user.setUpdatedAt(LocalDateTime.now());

        User updatedUser = userRepository.save(user);
        log.info("User details updated for: {}", user.getUsername());

        return userMapper.toResponseDto(updatedUser);
    }

    @Override
    public void deleteUser(String id){
        if(!userRepository.existsById(id)){
            throw new ResourceNotFoundException("Cannot delete: User not found with id: " + id);
        }

        userRepository.deleteById(id);
        log.info("User with id {} has been deleted.", id);
    }

}
