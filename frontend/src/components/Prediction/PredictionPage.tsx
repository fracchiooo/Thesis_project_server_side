import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";
import PredictionCard from "./PredictionCard.tsx";
import { FaPlus, FaDatabase } from "react-icons/fa";
import { Prediction } from "../../types/Prediction.ts";

const PredictionPage = () => {
    const [predictions, setPredictions] = useState<Prediction[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
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
    };

    const handleCreate = async () => {
        if (token == null) throw new Error("the token is null");
        const parsedData = JSON.parse(token);
        const tokenParsed = parsedData.token;

        const predictionDto = {
            initialConcentration: Number(newPrediction.initialConcentration),
            frequency: Number(newPrediction.frequency),
            dutyCycle: Number(newPrediction.dutyCycle),
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
        } catch (err: any) {
            setError(err.message);
            alert('Failed to create prediction: ' + err.message);
        } finally {
            setLoading(false);
            fetchPredictions();
        }

        setIsModalOpen(false);
        setNewPrediction({
            initialConcentration: '',
            frequency: '',
            dutyCycle: '',
            timeLasted: '',
            temperature: ''
        });
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
            
            for (const prediction of selectedData) {
                // Verifica che l'ID esista
                if (!prediction.id) {
                    console.error('Prediction without ID:', prediction);
                    continue;
                }

                const dataDto = {
                    id: prediction.id.valueOf(),  // ← AGGIUNGI L'ID
                    initialConcentration: prediction.initialConcentration?.valueOf() || 0,
                    frequency: prediction.frequency?.valueOf() || 0,
                    dutyCycle: prediction.dutyCycle?.valueOf() || 0,
                    timeLasted: prediction.timeLasted?.valueOf() || 0,
                    temperature: prediction.temperature?.valueOf() || 0,
                    observedConcentration: prediction.observedConcentration?.valueOf() || 0
                };

                console.log('Sending dataDto:', dataDto);

                await axios.post(
                    "http://localhost:3000/prediction/addData",
                    dataDto,
                    {
                        headers: {
                            Authorization: `Bearer ${tokenParsed}`,
                            'Content-Type': 'application/json'
                        }
                    }
                );
                successCount++;
            }

            alert(`Successfully added ${successCount} prediction(s) to dataset!`);
            setSelectedPredictions(new Set());
            await fetchPredictions(); // Ricarica per aggiornare sentToDataset
        } catch (err: any) {
            console.error('Error adding to dataset:', err);
            setError(err.message);
            alert('Failed to add to dataset: ' + (err.response?.data || err.message));
        } finally {
            setLoading(false);
        }
    };

    if (loading) return <div className="loading">Loading...</div>;
    if (error) return <div className="error">Error: {error}</div>;

    return (
        <div className="shopping-list-page-wrapper">
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

            <div className="shopping-lists-container">
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
                    <div className="no-shopping-lists">No predictions found.</div>
                )}
            </div>

            {isModalOpen && (
                <div className="modal">
                    <div className="modal-content">
                        <h2>Create New Prediction</h2>
                        
                        <div className="form-group">
                            <label>Initial Concentration:</label>
                            <input
                                type="number"
                                step="0.01"
                                value={newPrediction.initialConcentration}
                                onChange={(e) => handleInputChange('initialConcentration', e.target.value)}
                                placeholder="Enter initial concentration"
                            />
                        </div>

                        <div className="form-group">
                            <label>Frequency:</label>
                            <input
                                type="number"
                                step="0.01"
                                value={newPrediction.frequency}
                                onChange={(e) => handleInputChange('frequency', e.target.value)}
                                placeholder="Enter frequency"
                            />
                        </div>

                        <div className="form-group">
                            <label>Duty Cycle:</label>
                            <input
                                type="number"
                                step="0.01"
                                value={newPrediction.dutyCycle}
                                onChange={(e) => handleInputChange('dutyCycle', e.target.value)}
                                placeholder="Enter duty cycle"
                            />
                        </div>

                        <div className="form-group">
                            <label>Time Lasted:</label>
                            <input
                                type="number"
                                step="0.01"
                                value={newPrediction.timeLasted}
                                onChange={(e) => handleInputChange('timeLasted', e.target.value)}
                                placeholder="Enter time lasted"
                            />
                        </div>

                        <div className="form-group">
                            <label>Temperature:</label>
                            <input
                                type="number"
                                step="0.01"
                                value={newPrediction.temperature}
                                onChange={(e) => handleInputChange('temperature', e.target.value)}
                                placeholder="Enter temperature"
                            />
                        </div>

                        <div className="button-group">
                            <button onClick={handleCreate}>Create</button>
                            <button onClick={handleCancel}>Cancel</button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default PredictionPage;