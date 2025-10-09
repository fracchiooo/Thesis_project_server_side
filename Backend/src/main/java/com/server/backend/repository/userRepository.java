package com.server.backend.repository;

import java.util.Optional;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import com.server.backend.model.User;

@Repository
public interface userRepository extends JpaRepository<User, String> {
    
    Optional<User> findByUsernameAndHashPassword(String username, String hash_pass);

    Optional<User> findOneByUsername(String username);
}
