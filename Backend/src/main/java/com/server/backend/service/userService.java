package com.server.backend.service;

import java.time.Instant;
import java.util.Base64;
import java.util.Date;
import java.util.Optional;

import javax.crypto.SecretKey;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.stereotype.Service;
import org.springframework.web.server.ResponseStatusException;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.server.backend.dto.loginDto;
import com.server.backend.dto.userProfileRequestDto;
import com.server.backend.dto.userProfileResponseDto;
import com.server.backend.model.User;
import com.server.backend.repository.userRepository;

import io.jsonwebtoken.Jwts;
import io.jsonwebtoken.security.Keys;

@Service
public class userService {

    @Value("${authentication.service.jwtSecret}")
    private String secretKey;
    @Value("${authentication.service.jwtExpirationTime}")
    private long expirationTime;

    @Autowired
    private ObjectMapper objectMapper;

    @Autowired
    private userRepository userRepo;

    private final BCryptPasswordEncoder passwordEncoder = new BCryptPasswordEncoder();

    private String hashPassword(String plainPassword) {
        return passwordEncoder.encode(plainPassword);
    }

    private boolean verifyPassword(String plainPassword, String hashedPassword) {
        return passwordEncoder.matches(plainPassword, hashedPassword);
    }

    
    public userProfileResponseDto createUserProfile(userProfileRequestDto dto) throws Exception{
        User u = new User();
        u.setUsername(dto.getUsername());
        u.setHashPassword(hashPassword(dto.getPassword()));

        if (userRepo.findOneByUsername(dto.getUsername()).isPresent()) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "Profile already created for user " + dto.getUsername());
        }
        u = userRepo.save(u);
        userRepo.flush();
        return objectMapper.convertValue(u, userProfileResponseDto.class);
    }

    public ResponseEntity<Object> login(loginDto dto){
        Optional<User> up = userRepo.findOneByUsername(dto.getUsername());

        if (up.isEmpty()) return new ResponseEntity<>("username not present", HttpStatus.NOT_FOUND);
        if(!verifyPassword(dto.getPassword(), up.get().getHashPassword())) return new ResponseEntity<>("the password is wrong", HttpStatus.FORBIDDEN);


        String username = up.get().getUsername();
        String jwt = generateToken(username);

        return new ResponseEntity<>(jwt, HttpStatus.OK);
    }



    private String generateToken(String username) {
        return Jwts.builder()
                .setSubject(username)
                .setIssuedAt(new Date())
                .setExpiration(new Date(System.currentTimeMillis() + expirationTime))
                .signWith(generateKey())
                .compact();
    }

    private SecretKey generateKey() {
        System.out.println(secretKey);
        byte[] decodeKey = Base64.getDecoder().decode(secretKey);
        return Keys.hmacShaKeyFor(decodeKey);
    }

    public boolean validateToken(String token) {
        try {
            boolean ret = Jwts.parserBuilder().setSigningKey(generateKey()).build().parseClaimsJws(token).getBody().getExpiration().after(Date.from(Instant.now()));
            return ret;
        } catch (Exception e) {
            return false;
        }
    }

    public String getUsernameFromToken(String token) {
        try {
            return Jwts.parserBuilder().setSigningKey(generateKey()).build()
            .parseClaimsJws(token)
            .getBody().getSubject();
        } catch (Exception e) {
            return null;
        }
    }

    
}
