package com.anomaly.detection.backend.Controller;

import com.anomaly.detection.backend.Dto.UserResponseDto;
import com.anomaly.detection.backend.Model.User;
import com.anomaly.detection.backend.Service.UserService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/users")
@RequiredArgsConstructor
@Slf4j
@CrossOrigin(origins = "*")
public class UserController {

    private final UserService userService;

    @PostMapping("/register")
    public ResponseEntity<UserResponseDto> register(@RequestBody User user){
        log.info("REST request to register user: {}", user.getUsername());
        UserResponseDto response = userService.registerUser(user);

        return new ResponseEntity<>(response, HttpStatus.CREATED);
    }

    @GetMapping
    public ResponseEntity<List<UserResponseDto>> getAllUsers(){
        log.info("REST request to get all users");
        List<UserResponseDto> users = userService.getAllUsers();
        return ResponseEntity.ok(users);
    }

    @GetMapping("/{id}")
    public ResponseEntity<UserResponseDto> getUserById(@PathVariable String id){
        log.info("REST request to get user by id: {}", id);
        UserResponseDto user = userService.getUserById(id);
        return ResponseEntity.ok(user);
    }

    @PutMapping("/{id}")
    public ResponseEntity<UserResponseDto> updateUser(@PathVariable String id, @RequestBody User userDetails){
        log.info("REST request to update user: {}", id);
        UserResponseDto updatedUser = userService.updateUser(id, userDetails);
        return ResponseEntity.ok(updatedUser);
    }

    @DeleteMapping("/{id}")
    public ResponseEntity<Void> deleteUser(@PathVariable String id){
        log.info("REST request to delete user: {}", id);
        userService.deleteUser(id);
        return ResponseEntity.noContent().build();
    }
}
