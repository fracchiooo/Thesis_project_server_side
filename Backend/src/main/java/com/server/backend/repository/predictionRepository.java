package com.server.backend.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import com.server.backend.model.Prediction;

@Repository
public interface predictionRepository extends JpaRepository<Prediction, Long> {
    
}
