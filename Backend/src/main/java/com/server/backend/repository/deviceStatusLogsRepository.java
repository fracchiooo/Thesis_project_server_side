package com.server.backend.repository;

import java.util.Date;
import java.util.List;

import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import com.server.backend.model.DeviceStatusLogs;

@Repository
public interface deviceStatusLogsRepository extends JpaRepository<DeviceStatusLogs, Long> {

    List<DeviceStatusLogs> findByDeviceDeviceEUI(String deviceEUI);

    Page<DeviceStatusLogs> findByDeviceDeviceEUIAndStatusDateBetween(
        String deviceEUI, 
        Date startDate, 
        Date endDate, 
        Pageable pageable
    );
    
    
}
