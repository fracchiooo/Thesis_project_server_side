package com.server.backend.controller;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.server.backend.dto.loginDto;
import com.server.backend.dto.userProfileRequestDto;
import com.server.backend.dto.userProfileResponseDto;
import com.server.backend.service.userService;

@RestController
@RequestMapping("/user")
public class userController {

    @Autowired
    private userService userServ;


    @PostMapping("/create")
    public ResponseEntity<Object> createUser(@RequestBody userProfileRequestDto userProfileRequestDto) {
        try {
            userProfileResponseDto createdProfile = userServ.createUserProfile(userProfileRequestDto);
            return new ResponseEntity<>(createdProfile, HttpStatus.CREATED);
        } catch (Exception e) {
            return new ResponseEntity<>(e.getMessage(), HttpStatus.NOT_ACCEPTABLE);
        }
    }

    @PostMapping("/login")
    public ResponseEntity<Object> login(@RequestBody loginDto loginDto) {
        System.out.println("sono nel controller, helo\n\n");
         return userServ.login(loginDto);
    }

    
}
