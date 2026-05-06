package com.anomaly.detection.backend.Controller;

import com.anomaly.detection.backend.Dto.AuthResponseDto;
import com.anomaly.detection.backend.Dto.UserLoginDto;
import com.anomaly.detection.backend.Security.JwtUtil;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.core.userdetails.UserDetailsService;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/auth")
@RequiredArgsConstructor
public class AuthController {

    private final AuthenticationManager authenticationManager;
    private final UserDetailsService userDetailsService;
    private final JwtUtil jwtUtil;

    @PostMapping("/login")
    public ResponseEntity<AuthResponseDto> login(@RequestBody UserLoginDto loginDto) {
        authenticationManager.authenticate(
                new UsernamePasswordAuthenticationToken(loginDto.getUsername(), loginDto.getPassword())
        );

        final UserDetails userDetails = userDetailsService.loadUserByUsername(loginDto.getUsername());

        final String jwt = jwtUtil.generateToken(userDetails.getUsername());

        String role = userDetails.getAuthorities().stream()
                .findFirst()
                .map(auth -> auth.getAuthority())
                .orElse("ROLE_USER");

        return ResponseEntity.ok(new AuthResponseDto(jwt, userDetails.getUsername(), role));
    }
}
