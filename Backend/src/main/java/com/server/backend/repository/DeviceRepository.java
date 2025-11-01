package com.server.backend.repository;

import java.util.List;
import java.util.Optional;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import com.server.backend.model.Device;

@Repository
public interface DeviceRepository extends JpaRepository<Device, String> {

    List<Device> findByUserUsername(String username);
    
    Optional<Device> findByDeviceEUIAndUserUsername(String deviceEUI, String username);
}
