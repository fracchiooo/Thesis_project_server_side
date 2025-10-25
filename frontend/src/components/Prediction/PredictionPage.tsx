import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";
import PredictionCard from "./PredictionCard.tsx";
import { FaPlus, FaDatabase, FaBrain } from "react-icons/fa";
import { Prediction } from "../../types/Prediction.ts";
import '../../CSS/prediction.css';

const PredictionPage = () => {
    const [predictions, setPredictions] = useState<Prediction[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [isTraining, setIsTraining] = useState(false);
    const token = localStorage.getItem('authToken');
    const navigate = useNavigate();
    const [isModalOpen, setIsModalOpen] = useState(false);
    const [selectedPredictions, setSelectedPredictions] = useState<Set<number>>(new Set());
    const [newPrediction, setNewPrediction] = useState({
        initialConcentration: '',
        frequency: '',
        dutyCycle: '',
        timeLasted: '',
        temperature: ''
    });
    const [validationErrors, setValidationErrors] = useState<string[]>([]);

    const fetchPredictions = async () => {
        if (token == null) {
            navigate('/login');
            return;
        }
        const parsedData = JSON.parse(token);
        const tokenParsed = parsedData.token;
        try {
            setLoading(true);
            const response = await axios.get("http://localhost:3000/prediction/list", {
                headers: {
                    Authorization: `Bearer ${tokenParsed}`
                }
            });
            const data = response.data;
            setPredictions(data);
            console.log(data);
        } catch (err: any) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchPredictions();
    }, []);

    const onCreate = () => {
        setIsModalOpen(true);
        setValidationErrors([]);
    };

    const validatePrediction = (): boolean => {
        const errors: string[] = [];

        if (!newPrediction.initialConcentration || !newPrediction.frequency || 
            !newPrediction.dutyCycle || !newPrediction.timeLasted || !newPrediction.temperature) {
            errors.push('All fields are required');
            setValidationErrors(errors);
            return false;
        }

        const frequency = parseFloat(newPrediction.frequency);
        const dutyCycle = parseFloat(newPrediction.dutyCycle);
        const temperature = parseFloat(newPrediction.temperature);
        const initialConcentration = parseFloat(newPrediction.initialConcentration);
        const timeLasted = parseFloat(newPrediction.timeLasted);

        if (isNaN(frequency) || frequency < 0 || frequency > 40) {
            errors.push('Frequency must be between 0 and 40 Hz');
        }

        if (isNaN(dutyCycle) || dutyCycle < 0 || dutyCycle > 100) {
            errors.push('Duty Cycle must be between 0 and 100%');
        }

        if (isNaN(temperature) || temperature < 18 || temperature > 28) {
            errors.push('Temperature must be between 18 and 28°C');
        }

        if (isNaN(initialConcentration) || initialConcentration <= 0) {
            errors.push('Initial Concentration must be greater than 0');
        }

        if (isNaN(timeLasted) || timeLasted <= 0) {
            errors.push('Time Lasted must be greater than 0');
        }

        setValidationErrors(errors);
        return errors.length === 0;
    };

    const handleCreate = async () => {
        if (token == null) throw new Error("the token is null");

        if (!validatePrediction()) {
            return;
        }

        const parsedData = JSON.parse(token);
        const tokenParsed = parsedData.token;

        const predictionDto = {
            initialConcentration: Number(newPrediction.initialConcentration),
            frequency: Number(newPrediction.frequency),
            dutyCycle: Number(newPrediction.dutyCycle) / 100,
            timeLasted: Number(newPrediction.timeLasted),
            temperature: Number(newPrediction.temperature)
        };

        try {
            setLoading(true);
            await axios.post("http://localhost:3000/prediction/predict", predictionDto, {
                headers: {
                    Authorization: `Bearer ${tokenParsed}`,
                    'Content-Type': 'application/json'
                }
            });
            alert('Prediction created successfully!');
            setIsModalOpen(false);
            setNewPrediction({
                initialConcentration: '',
                frequency: '',
                dutyCycle: '',
                timeLasted: '',
                temperature: ''
            });
            setValidationErrors([]);
        } catch (err: any) {
            setError(err.message);
            alert('Failed to create prediction: ' + err.message);
        } finally {
            setLoading(false);
            fetchPredictions();
        }
    };

    const handleCancel = () => {
        setIsModalOpen(false);
        setNewPrediction({
            initialConcentration: '',
            frequency: '',
            dutyCycle: '',
            timeLasted: '',
            temperature: ''
        });
        setValidationErrors([]);
    };

    const handleInputChange = (field: string, value: string) => {
        setNewPrediction(prev => ({
            ...prev,
            [field]: value
        }));
    };

    const handleSelectPrediction = (id: number, isSelected: boolean) => {
        setSelectedPredictions(prev => {
            const newSet = new Set(prev);
            if (isSelected) {
                newSet.add(id);
            } else {
                newSet.delete(id);
            }
            return newSet;
        });
    };

    const completedPredictions = predictions.filter(p => 
        p.observedConcentration != null && !p.sentToDataset
    );

    const handleAddToDataset = async () => {
        if (selectedPredictions.size === 0) {
            alert('Please select at least one prediction');
            return;
        }

        if (token == null) throw new Error("Token is null");
        const parsedData = JSON.parse(token);
        const tokenParsed = parsedData.token;

        const selectedData = predictions.filter(p => {
            const predId = p.id?.valueOf();
            return predId != null && selectedPredictions.has(predId);
        });

        console.log('Selected data:', selectedData);

        try {
            setLoading(true);
            let successCount = 0;

            for (const pred of selectedData) {
                try {
                    const datasetDto = {
                        id: pred.id,
                        initialConcentration: pred.initialConcentration,
                        frequency: pred.frequency,
                        dutyCycle: pred.dutyCycle,
                        timeLasted: pred.timeLasted,
                        temperature: pred.temperature,
                        observedConcentration: pred.observedConcentration
                    };

                    await axios.post("http://localhost:3000/prediction/addData", datasetDto, {
                        headers: {
                            Authorization: `Bearer ${tokenParsed}`,
                            'Content-Type': 'application/json'
                        }
                    });

                    successCount++;
                } catch (err: any) {
                    console.error(`Error adding prediction ${pred.id} to dataset:`, err);
                }
            }

            alert(`Successfully added ${successCount} prediction(s) to dataset!`);
            setSelectedPredictions(new Set());
            await fetchPredictions();
        } catch (err: any) {
            console.error('Error adding to dataset:', err);
            setError(err.message);
            alert('Failed to add to dataset: ' + (err.response?.data || err.message));
        } finally {
            setLoading(false);
        }
    };

    const handleTrain = async () => {
        if (!window.confirm('Are you sure you want to train the model? This may take several minutes.')) {
            return;
        }

        if (token == null) throw new Error("Token is null");
        const parsedData = JSON.parse(token);
        const tokenParsed = parsedData.token;

        try {
            setIsTraining(true);
            const response = await axios.post(
                "http://localhost:3000/prediction/train",
                {},
                {
                    headers: {
                        Authorization: `Bearer ${tokenParsed}`
                    }
                }
            );
            alert('Model training completed successfully!');
            console.log('Training response:', response.data);
        } catch (err: any) {
            console.error('Error training model:', err);
            alert('Failed to train model: ' + (err.response?.data || err.message));
        } finally {
            setIsTraining(false);
        }
    };

    if (loading) return <div className="loading">Loading...</div>;
    if (error) return <div className="error">Error: {error}</div>;

    return (
        <div className="prediction-page-wrapper">
            <div className="page-header">
                <button className="create-button" onClick={onCreate}>
                    <FaPlus /> Create New Prediction
                </button>
                
                {completedPredictions.length > 0 && (
                    <button 
                        className="dataset-button" 
                        onClick={handleAddToDataset}
                        disabled={selectedPredictions.size === 0}
                    >
                        <FaDatabase /> Add to Dataset ({selectedPredictions.size})
                    </button>
                )}
            </div>

            <div className="predictions-container">
                {predictions.length > 0 ? (
                    predictions.map((prediction, index) => (
                        <PredictionCard 
                            key={prediction.id || index} 
                            prediction={prediction}
                            onUpdate={fetchPredictions}
                            isSelected={selectedPredictions.has(prediction.id?.valueOf() || 0)}
                            onSelect={handleSelectPrediction}
                            isSelectable={prediction.observedConcentration != null && !prediction.sentToDataset}
                        />
                    ))
                ) : (
                    <div className="no-predictions">No predictions found.</div>
                )}
            </div>

            {/* Train Model Button */}
            <div className="train-section">
                <button 
                    className="train-button" 
                    onClick={handleTrain}
                    disabled={isTraining}
                >
                    {isTraining ? (
                        <>
                            <span className="spinner"></span> Training Model...
                        </>
                    ) : (
                        <>
                            <FaBrain /> Train Model
                        </>
                    )}
                </button>
                <p className="train-info">
                    Train the prediction model with the current dataset to improve accuracy
                </p>
            </div>

            {isModalOpen && (
                <div className="modal-overlay" onClick={handleCancel}>
                    <div className="modal-content" onClick={(e) => e.stopPropagation()}>
                        <h2>Create New Prediction</h2>
                        
                        {validationErrors.length > 0 && (
                            <div className="validation-errors">
                                {validationErrors.map((error, index) => (
                                    <div key={index} className="validation-error">
                                        ⚠️ {error}
                                    </div>
                                ))}
                            </div>
                        )}
                        
                        <div className="form-group">
                            <label>
                                Initial Concentration:
                                <span className="range-hint">(&gt; 0)</span>
                            </label>
                            <input
                                type="number"
                                step="0.01"
                                min="0.01"
                                value={newPrediction.initialConcentration}
                                onChange={(e) => handleInputChange('initialConcentration', e.target.value)}
                                placeholder="Enter initial concentration"
                            />
                        </div>

                        <div className="form-group">
                            <label>
                                Frequency (Hz):
                                <span className="range-hint">(0 - 40)</span>
                            </label>
                            <input
                                type="number"
                                step="0.01"
                                min="0"
                                max="40"
                                value={newPrediction.frequency}
                                onChange={(e) => handleInputChange('frequency', e.target.value)}
                                placeholder="Enter frequency (0-40 Hz)"
                            />
                        </div>

                        <div className="form-group">
                            <label>
                                Duty Cycle (%):
                                <span className="range-hint">(0 - 100)</span>
                            </label>
                            <input
                                type="number"
                                step="0.01"
                                min="0"
                                max="100"
                                value={newPrediction.dutyCycle}
                                onChange={(e) => handleInputChange('dutyCycle', e.target.value)}
                                placeholder="Enter duty cycle (0-100%)"
                            />
                        </div>

                      <div className="form-group">
                            <label>
                                Time Lasted (hours):
                                <span className="range-hint">(&gt; 0)</span>
                            </label>
                            <input
                                type="number"
                                step="0.01"
                                min="0.01"
                                value={newPrediction.timeLasted}
                                onChange={(e) => handleInputChange('timeLasted', e.target.value)}
                                placeholder="Enter time lasted"
                            />
                        </div>
                        <div className="form-group">
                            <label>
                                Temperature (°C):
                                <span className="range-hint">(18 - 28)</span>
                            </label>
                            <input
                                type="number"
                                step="0.01"
                                min="18"
                                max="28"
                                value={newPrediction.temperature}
                                onChange={(e) => handleInputChange('temperature', e.target.value)}
                                placeholder="Enter temperature (18-28°C)"
                            />
                        </div>

                        <div className="button-group">
                            <button className="primary-button" onClick={handleCreate}>Create</button>
                            <button className="secondary-button" onClick={handleCancel}>Cancel</button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default PredictionPage;