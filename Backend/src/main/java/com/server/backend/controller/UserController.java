package com.server.backend.controller;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.server.backend.dto.LoginDto;
import com.server.backend.dto.UserProfileRequestDto;
import com.server.backend.dto.UserProfileResponseDto;
import com.server.backend.service.UserService;

@RestController
@RequestMapping("/user")
public class UserController {

    @Autowired
    private UserService userServ;

    @PostMapping("/create")
    public ResponseEntity<Object> createUser(@RequestBody UserProfileRequestDto userProfileRequestDto) {
        try {
            UserProfileResponseDto createdProfile = userServ.createUserProfile(userProfileRequestDto);
            return new ResponseEntity<>(createdProfile, HttpStatus.CREATED);
        } catch (Exception e) {
            return new ResponseEntity<>(e.getMessage(), HttpStatus.NOT_ACCEPTABLE);
        }
    }

    @PostMapping("/login")
    public ResponseEntity<Object> login(@RequestBody LoginDto loginDto) {
         return userServ.login(loginDto);
    } 
}
