package com.server.backend.model;

import java.util.Date;
import java.util.Map;

import com.fasterxml.jackson.annotation.JsonIgnore;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.JoinColumn;
import jakarta.persistence.ManyToOne;
import jakarta.persistence.Table;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;


@NoArgsConstructor
@AllArgsConstructor
@Data
@Entity
@Table(name = "prediction")
public class Prediction {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(name = "timestamp")
    private Date timestamp;

    @Column(name = "predicted_concentration")
    private Float predictedConcentration;

    @Column(name = "predicted_uncertainty")
    private Float predictedUncertainty;

    @Column(name = "observed_concentration")
    private Float observedConcentration;

    @Column(name = "initial_concentration")
    private Float initialConcentration;

    @Column(name = "frequency")
    private Float frequency;
    
    @Column(name = "duty_cycle")
    private Float dutyCycle;

    @Column(name = "time_lasted")
    private Float timeLasted;

    @Column(name = "temperature")
    private Float temperature;

    @JsonIgnore
    @ManyToOne
    @JoinColumn(name = "user", nullable = true)
    private User user;


    public void setEnvironmentalConditions(float frequency, Float dutyCycle, Float temperature) {
        this.frequency = frequency;
        this.dutyCycle = dutyCycle;
        this.temperature = temperature;
    }


    
}
