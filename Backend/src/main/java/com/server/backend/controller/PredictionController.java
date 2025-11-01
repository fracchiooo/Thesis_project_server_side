package com.server.backend.controller;

import java.util.List;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.server.backend.dto.CompletePredictionDto;
import com.server.backend.dto.DataForDatasetDto;
import com.server.backend.dto.PredictionDto;
import com.server.backend.model.Prediction;
import com.server.backend.service.PredictionService;


@RestController
@RequestMapping("/prediction")
public class PredictionController {

    @Autowired
    private PredictionService predictionServ;

    @PostMapping("/predict")
    public ResponseEntity<Prediction> predict(@RequestBody PredictionDto predDto) {
        return predictionServ.predict(predDto);
        
    }

    @PutMapping("predict/{id}")
    public ResponseEntity<Object> completePrediction(@PathVariable Long id, @RequestBody CompletePredictionDto observed) {
        return predictionServ.completePrediction(id, observed);
    }


    @GetMapping("/list")
    public ResponseEntity<List<Prediction>> listPredictions() {
        return predictionServ.listPredictions();
    }
    

    @PostMapping("/train")
    public ResponseEntity<Object> train() {
        return predictionServ.train();
        
    }

    @PostMapping("/addData")
    public ResponseEntity<Object> addData(@RequestBody DataForDatasetDto dataDto) {
        return predictionServ.addData(dataDto);
    }
}
