package com.server.backend.service;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpMethod;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import com.server.backend.dto.CompletePredictionDto;
import com.server.backend.dto.DataForDatasetDto;
import com.server.backend.dto.ModelResponseDto;
import com.server.backend.dto.PredictResponseDto;
import com.server.backend.dto.PredictionDto;
import com.server.backend.model.Prediction;
import com.server.backend.repository.predictionRepository;
import com.server.backend.repository.userRepository;
import com.server.backend.utilities.JWTContext;

@Service
public class predictionService {

    @Autowired
    private predictionRepository predictionRepo;

    @Autowired
    private userRepository userRepo;
    
    @Autowired
    private RestTemplate restTemplate;

    @Value("${spring.model.host}")
    private String modelHost;

    @Value("${spring.model.port}")
    private String modelPort;

    @Value("${spring.model.api_key}")
    private String apiKey;


    public ResponseEntity<PredictResponseDto> predict(PredictionDto predDto) {
        // TODO make call to the model microservice
        // TODO save the prediction in the database (without observed data)
        
        HttpHeaders headers = new HttpHeaders();
        headers.set("Content-Type", "application/json");
        headers.set("Authorization", apiKey);

        // Wrap the request payload in an HttpEntity with headers
        HttpEntity<Object> requestEntity = new HttpEntity<>(predDto, headers);

        String predictUrl = "http://" + modelHost + ":" + modelPort + "/predict";
        // Make the POST request
        ResponseEntity<ModelResponseDto> res = restTemplate.exchange(predictUrl, HttpMethod.POST, requestEntity, ModelResponseDto.class);

        if (res.getStatusCode().is2xxSuccessful() && res.getBody() != null) {
            ModelResponseDto responseBody = res.getBody();
            if (responseBody!=null && responseBody.getStatus().equals("success")) {
                // Extract the 'result' field which contains the prediction details
                Map<String, Object> result = (Map<String, Object>) responseBody.getResult();
                Map<String, Object> gompertz_pred = (Map<String, Object>) result.get("gompertz");
                
                Prediction p = new Prediction();
                p.setTimestamp(new java.util.Date());
                p.setPredictedConcentration(Float.valueOf(gompertz_pred.get("mean").toString()));
                p.setPredictedUncertainty(Float.valueOf(gompertz_pred.get("std").toString()));
                p.setInitialConcentration(predDto.getInitialConcentration());
                p.setTimeLasted(predDto.getTimeLasted());
                p.setEnvironmentalConditions(predDto.getFrequency(), predDto.getDutyCycle(), predDto.getTemperature());

                String username = JWTContext.get();
                p.setUser(userRepo.findOneByUsername(username).orElse(null));
                
                p = predictionRepo.save(p);
                predictionRepo.flush();

                PredictResponseDto responseDto = new PredictResponseDto();
                responseDto.setId(p.getId());
                responseDto.setTimestamp(p.getTimestamp());
                responseDto.setPredictedConcentration(p.getPredictedConcentration());
                responseDto.setPredictedUncertainty(p.getPredictedUncertainty());
                responseDto.setInitialConcentration(p.getInitialConcentration());
                responseDto.setFrequency(p.getFrequency());
                responseDto.setDutyCycle(p.getDutyCycle());
                responseDto.setTimeLasted(p.getTimeLasted());
                responseDto.setTemperature(p.getTemperature());

                return ResponseEntity.ok(responseDto);
            } else {
                return ResponseEntity.status(500).body(null);
            }
        } else {
            return ResponseEntity.status(res.getStatusCode()).body(null);
        }
    }



    public ResponseEntity<Object> completePrediction(Long id, CompletePredictionDto observed) {
        Prediction p = predictionRepo.findById(id).orElse(null);
        if (p == null) {
            return ResponseEntity.status(404).body("Prediction not found");
        }
        p.setObservedConcentration(observed.getObserved_density());
        predictionRepo.save(p);
        predictionRepo.flush();
        return ResponseEntity.ok("Prediction updated");
    }



    public ResponseEntity<List<Prediction>> listPredictions() {

        List<Prediction> predictions = predictionRepo.findAll();

        if(predictions.isEmpty()) {
            return ResponseEntity.status(404).body(null);
        }

        return ResponseEntity.ok(predictions);
    }



    public ResponseEntity<Object> train() {

        HttpHeaders headers = new HttpHeaders();
        headers.set("Content-Type", "application/json");
        headers.set("Authorization", apiKey);

        // Wrap the request payload in an HttpEntity with headers
        HttpEntity<Object> requestEntity = new HttpEntity<>(new HashMap<>(), headers);

        String predictUrl = "http://" + modelHost + ":" + modelPort + "/train";
        // Make the POST request
        ResponseEntity<ModelResponseDto> res = restTemplate.exchange(predictUrl, HttpMethod.POST, requestEntity, ModelResponseDto.class);

        if (res.getStatusCode().is2xxSuccessful() && res.getBody() != null) {
            ModelResponseDto responseBody = res.getBody();
            if (responseBody!=null && responseBody.getStatus().equals("success")) {
                return ResponseEntity.ok(responseBody.getResult());
            } else {
                return ResponseEntity.status(500).body("Successful response but error in decoding response body");
            }
        } else {
            return ResponseEntity.status(res.getStatusCode()).body(res.getBody() != null && res.getBody().getResult() != null ? res.getBody().getResult() : "Error during training");
        }
    }


    public ResponseEntity<Object> addData(DataForDatasetDto dataDto) {

        HttpHeaders headers = new HttpHeaders();
        headers.set("Content-Type", "application/json");
        headers.set("Authorization", apiKey);

        // Wrap the request payload in an HttpEntity with headers
        HttpEntity<Object> requestEntity = new HttpEntity<>(dataDto, headers);

        String predictUrl = "http://" + modelHost + ":" + modelPort + "/addData";
        // Make the POST request
        ResponseEntity<ModelResponseDto> res = restTemplate.exchange(predictUrl, HttpMethod.POST, requestEntity, ModelResponseDto.class);
        
        if (res.getStatusCode().is2xxSuccessful() && res.getBody() != null) {
            ModelResponseDto responseBody = res.getBody();
            if (responseBody!=null && responseBody.getStatus().equals("success")) {
                return ResponseEntity.ok(responseBody.getResult());
            } else {
                return ResponseEntity.status(500).body("Successful response but error in decoding response body");
            }
        } else {
            return ResponseEntity.status(res.getStatusCode()).body(res.getBody() != null && res.getBody().getResult() != null ? res.getBody().getResult() : "Error during adding data");
        }
    }


    
}
