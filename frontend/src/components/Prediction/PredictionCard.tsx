import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";
import { Prediction } from "../../types/Prediction";
import { FaThermometerHalf, FaClock, FaFlask, FaBroadcastTower, FaEdit, FaSave, FaTimes } from "react-icons/fa";
import '../../CSS/prediction.css';

interface PredictionCardProps {
    prediction: Prediction;
    onUpdate?: () => void;
    isSelected?: boolean;
    onSelect?: (id: number, isSelected: boolean) => void;
    isSelectable?: boolean;
}

const PredictionCard: React.FC<PredictionCardProps> = ({ 
    prediction, 
    onUpdate, 
    isSelected = false, 
    onSelect,
    isSelectable = false,
}) => {
    const navigate = useNavigate();
    const [isEditing, setIsEditing] = useState(false);
    const [observedValue, setObservedValue] = useState('');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    
    const formatDate = (timestamp?: Date | string) => {
        if (!timestamp) return 'N/A';
        try {
            const date = typeof timestamp === 'string' ? new Date(timestamp) : timestamp;
            return date.toLocaleString();
        } catch {
            return 'Invalid Date';
        }
    };

    const formatNumber = (num?: Number | null) => {
        return num != null ? num.valueOf().toFixed(2) : 'N/A';
    };

    const handleEdit = () => {
        setIsEditing(true);
        setError(null);
    };

    const handleCancel = () => {
        setIsEditing(false);
        setObservedValue('');
        setError(null);
    };

    const handleSave = async () => {
        if (!observedValue.trim()) {
            setError('Please enter a value');
            return;
        }

        const token = localStorage.getItem('authToken');
        if (!token) {
            setError('Not authenticated');
            return;
        }

        const parsedData = JSON.parse(token);
        const tokenParsed = parsedData.token;

        try {
            setLoading(true);
            setError(null);

            const completePredictionDto = {
                observed_density: Number(observedValue)
            };

            await axios.put(
                `http://localhost:3000/prediction/predict/${prediction.id}`,
                completePredictionDto,
                {
                    headers: {
                        Authorization: `Bearer ${tokenParsed}`,
                        'Content-Type': 'application/json'
                    }
                }
            );

            alert('Observed concentration updated successfully!');
            setIsEditing(false);
            setObservedValue('');
            
            if (onUpdate) {
                onUpdate();
            }
        } catch (err: any) {
            setError(err.response?.data?.message || err.message || 'Failed to update');
            console.error('Error updating prediction:', err);
        } finally {
            setLoading(false);
        }
    };

    const handleCheckboxChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (onSelect && prediction.id) {
            onSelect(prediction.id.valueOf(), e.target.checked);
        }
    };

    return (
    <div className={`prediction-card ${isSelected ? 'selected' : ''} ${prediction.sentToDataset ? 'sent-to-dataset' : ''}`}>
        <div className="prediction-card-header">
            <div className="header-left">
                {isSelectable && (
                    <input
                        type="checkbox"
                        checked={isSelected}
                        onChange={handleCheckboxChange}
                        className="prediction-checkbox"
                        disabled={prediction.sentToDataset}
                    />
                )}
                <span className="prediction-id">Prediction #{prediction.id?.valueOf()}</span>
                {prediction.sentToDataset && (
                    <span style={{ 
                        marginLeft: '10px', 
                        padding: '4px 8px', 
                        backgroundColor: '#4CAF50', 
                        color: 'white', 
                        borderRadius: '4px', 
                        fontSize: '0.8em',
                        fontWeight: 'bold'
                    }}>
                        ✓ Sent to Dataset
                    </span>
                )}
            </div>
            <span className="prediction-timestamp">{formatDate(prediction.timestamp)}</span>
        </div>
        
        <div className="prediction-card-body">
            <div className="prediction-section">
                <h4 className="section-title">Predicted Results</h4>
                <div className="prediction-info-item">
                    <FaFlask className="info-icon" />
                    <span className="info-label">Predicted Concentration:</span>
                    <span className="info-value highlight">
                        {formatNumber(prediction.predictedConcentration)}
                    </span>
                </div>
                <div className="prediction-info-item">
                    <span className="info-label">Uncertainty:</span>
                    <span className="info-value">
                        ±{formatNumber(prediction.predictedUncertainty)}
                    </span>
                </div>

                <div className="prediction-info-item observed-section">
                    <span className="info-label">Observed Concentration:</span>
                    {prediction.observedConcentration != null ? (
                        <span className="info-value observed">
                            {formatNumber(prediction.observedConcentration)}
                        </span>
                    ) : isEditing ? (
                        <div className="edit-container">
                            <input
                                type="number"
                                step="0.01"
                                value={observedValue}
                                onChange={(e) => setObservedValue(e.target.value)}
                                placeholder="Enter observed value"
                                className="observed-input"
                                disabled={loading}
                            />
                            <button 
                                onClick={handleSave} 
                                className="save-button"
                                disabled={loading}
                                title="Save"
                            >
                                <FaSave />
                            </button>
                            <button 
                                onClick={handleCancel} 
                                className="cancel-button"
                                disabled={loading}
                                title="Cancel"
                            >
                                <FaTimes />
                            </button>
                        </div>
                    ) : (
                        <button 
                            onClick={handleEdit} 
                            className="edit-button"
                            title="Add observed concentration"
                        >
                            <FaEdit /> Add Value
                        </button>
                    )}
                </div>
                {error && <div className="error-message">{error}</div>}
            </div>

            <div className="prediction-section">
                <h4 className="section-title">Input Parameters</h4>
                <div className="prediction-info-item">
                    <span className="info-label">Initial Concentration:</span>
                    <span className="info-value">
                        {formatNumber(prediction.initialConcentration)}
                    </span>
                </div>
                <div className="prediction-info-item">
                    <FaBroadcastTower className="info-icon" />
                    <span className="info-label">Frequency:</span>
                    <span className="info-value">
                        {formatNumber(prediction.frequency)} Hz
                    </span>
                </div>
                <div className="prediction-info-item">
                    <span className="info-label">Duty Cycle:</span>
                    <span className="info-value">
                        {prediction.dutyCycle != null 
                            ? `${(prediction.dutyCycle.valueOf() * 100).toFixed(2)}%`
                            : 'N/A'}
                    </span>
                </div>
                <div className="prediction-info-item">
                    <FaClock className="info-icon" />
                    <span className="info-label">Time Lasted:</span>
                    <span className="info-value">
                        {formatNumber(prediction.timeLasted)} hours
                    </span>
                </div>
                <div className="prediction-info-item">
                    <FaThermometerHalf className="info-icon" />
                    <span className="info-label">Temperature:</span>
                    <span className="info-value">
                        {formatNumber(prediction.temperature)}°C
                    </span>
                </div>
            </div>
        </div>
    </div>
    );
};

export default PredictionCard;