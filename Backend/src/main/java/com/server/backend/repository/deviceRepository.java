package com.server.backend.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import com.server.backend.model.Device;

@Repository
public interface deviceRepository extends JpaRepository<Device, String> {
    
}
