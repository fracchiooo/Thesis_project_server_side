import React, { useState } from "react";
import { FaTrash, FaThermometerHalf, FaClock, FaBroadcastTower, FaPaperPlane } from "react-icons/fa";
import { useNavigate } from "react-router-dom";
import axios from "axios";
import { Device } from "../../types/Device";
import '../../CSS/device.css';

interface DeviceCardProps {
    device: Device;
    onDelete: (deviceEUI: string) => void;
}

const DeviceCard: React.FC<DeviceCardProps> = ({ device, onDelete }) => {
    const navigate = useNavigate();
    const [isCommandModalOpen, setIsCommandModalOpen] = useState(false);
    const [isSending, setIsSending] = useState(false);
    const [commandData, setCommandData] = useState({
        frequency: '',
        duty_frequency: '',
        finish_after: '',
        startTime: ''
    });
    const [validationErrors, setValidationErrors] = useState<string[]>([]);
    
    // Rimuovi virgolette extra dal deviceEUI se presenti
    const cleanDeviceEUI = device.deviceEUI?.replace(/^"|"$/g, '') || 'Unknown';
    
    // Formatta la data
    const formatDate = (date?: string) => {
        if (!date) return 'N/A';
        try {
            return new Date(date).toLocaleString();
        } catch {
            return 'Invalid Date';
        }
    };

    const handleCardClick = () => {
        navigate(`/device/${cleanDeviceEUI}`);
    };

    const handleSendCommand = (e: React.MouseEvent) => {
        e.stopPropagation();
        setIsCommandModalOpen(true);
        setValidationErrors([]);
    };

    const validateCommand = (): boolean => {
        const errors: string[] = [];

        // Validazione campi obbligatori
        if (!commandData.frequency || !commandData.duty_frequency || !commandData.finish_after) {
            errors.push('All required fields must be filled');
            setValidationErrors(errors);
            return false;
        }

        const frequency = parseFloat(commandData.frequency);
        const dutyFrequency = parseFloat(commandData.duty_frequency);
        const finishAfter = parseFloat(commandData.finish_after);

        // Validazione range frequency (0-40 Hz)
        if (isNaN(frequency) || frequency < 0 || frequency > 40) {
            errors.push('Frequency must be between 0 and 40 Hz');
        }

        // Validazione range duty_frequency (0-1)
        if (isNaN(dutyFrequency) || dutyFrequency < 20 || dutyFrequency > 100) {
            errors.push('Duty Frequency must be between 20 and 100');
        }

        // Validazione finish_after (maggiore di 0)
        if (isNaN(finishAfter) || finishAfter <= 0) {
            errors.push('Finish After must be greater than 0');
        }

        // Validazione startTime (opzionale, ma se presente deve essere futura)
        if (commandData.startTime) {
            const startTime = new Date(commandData.startTime);
            const now = new Date();
            if (startTime < now) {
                errors.push('Start Time must be in the future');
            }
        }

        setValidationErrors(errors);
        return errors.length === 0;
    };

    const handleCommandSubmit = async () => {
        const token = localStorage.getItem('authToken');
        if (!token) {
            alert('Not authenticated');
            return;
        }

        // Validazione
        if (!validateCommand()) {
            return;
        }

        const parsedData = JSON.parse(token);
        const tokenParsed = parsedData.token;

        try {
            setIsSending(true);

            const commandDto: any = {
                frequency: parseFloat(commandData.frequency),
                duty_frequency: parseFloat(commandData.duty_frequency),
                finish_after: parseFloat(commandData.finish_after)
            };

            // Aggiungi startTime solo se specificato
            if (commandData.startTime) {
                commandDto.startTime = new Date(commandData.startTime).toISOString();
            }

            await axios.post(
                `http://localhost:3000/device/sendCommand/${cleanDeviceEUI}`,
                commandDto,
                {
                    headers: {
                        Authorization: `Bearer ${tokenParsed}`,
                        'Content-Type': 'application/json'
                    }
                }
            );

            alert('Command sent successfully!');
            setIsCommandModalOpen(false);
            setCommandData({
                frequency: '',
                duty_frequency: '',
                finish_after: '',
                startTime: ''
            });
            setValidationErrors([]);
        } catch (err: any) {
            console.error('Error sending command:', err);
            alert('Failed to send command: ' + (err.response?.data || err.message));
        } finally {
            setIsSending(false);
        }
    };

    const handleCommandCancel = () => {
        setIsCommandModalOpen(false);
        setCommandData({
            frequency: '',
            duty_frequency: '',
            finish_after: '',
            startTime: ''
        });
        setValidationErrors([]);
    };

    return (
        <>
            <div className="device-card" onClick={handleCardClick} style={{ cursor: 'pointer' }}>
                <div className="device-card-header">
                    <span className="device-eui">
                        {cleanDeviceEUI}
                    </span>
                    <div className="device-actions">
                        <button 
                            className="command-button"
                            onClick={handleSendCommand}
                            title="Send command"
                        >
                            <FaPaperPlane/>
                        </button>
                        <button 
                            className="delete-button"
                            onClick={(e) => {
                                e.stopPropagation();
                                onDelete(device.deviceEUI);
                            }}
                            title="Delete device"
                        >
                            <FaTrash/>
                        </button>
                    </div>
                </div>
                
                <div className="device-card-body">
                    <div className="device-info-item">
                        <FaClock className="info-icon" />
                        <span className="info-label">Last Update:</span>
                        <span className="info-value">{formatDate(device.lastUpdate)}</span>
                    </div>
                    
                    <div className="device-info-item">
                        <FaThermometerHalf className="info-icon" />
                        <span className="info-label">Temperature:</span>
                        <span className="info-value">
                            {device.currentTemperature != null 
                                ? `${device.currentTemperature}°C` 
                                : 'N/A'}
                        </span>
                    </div>
                    
                    <div className="device-info-item">
                        <FaBroadcastTower className="info-icon" />
                        <span className="info-label">Frequency:</span>
                        <span className="info-value">
                            {device.currentSensedFrequency != null 
                                ? `${device.currentSensedFrequency} Hz` 
                                : 'N/A'}
                        </span>
                    </div>
                    
                    {device.logs && device.logs.length > 0 && (
                        <div className="device-info-item">
                            <span className="info-label">Logs:</span>
                            <span className="info-value">{device.logs.length} entries</span>
                        </div>
                    )}
                </div>
            </div>

            {/* Command Modal */}
            {isCommandModalOpen && (
                <div className="modal-overlay" onClick={handleCommandCancel}>
                    <div className="modal-content command-modal" onClick={(e) => e.stopPropagation()}>
                        <h2>Send Command to {cleanDeviceEUI}</h2>
                        
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
                                Frequency (Hz): <span className="required">*</span>
                                <span className="range-hint">(0 - 40)</span>
                            </label>
                            <input
                                type="number"
                                step="0.01"
                                min="0"
                                max="40"
                                value={commandData.frequency}
                                onChange={(e) => setCommandData(prev => ({ ...prev, frequency: e.target.value }))}
                                placeholder="Enter frequency (0-40 Hz)"
                                required
                            />
                        </div>

                        <div className="form-group">
                            <label>
                                Duty Frequency: <span className="required">*</span>
                                <span className="range-hint">(20 - 100)</span>
                            </label>
                            <input
                                type="number"
                                step="0.01"
                                min="0"
                                max="1"
                                value={commandData.duty_frequency}
                                onChange={(e) => setCommandData(prev => ({ ...prev, duty_frequency: e.target.value }))}
                                placeholder="Enter duty frequency (20-100)"
                                required
                            />
                        </div>

                        <div className="form-group">
                            <label>
                                Finish After (minutes): <span className="required">*</span>
                            </label>
                            <input
                                type="number"
                                step="0.01"
                                min="0.01"
                                value={commandData.finish_after}
                                onChange={(e) => setCommandData(prev => ({ ...prev, finish_after: e.target.value }))}
                                placeholder="Enter duration (minutes)"
                                required
                            />
                        </div>

                        <div className="form-group">
                            <label>Start Time (Optional):</label>
                            <input
                                type="datetime-local"
                                value={commandData.startTime}
                                onChange={(e) => setCommandData(prev => ({ ...prev, startTime: e.target.value }))}
                            />
                            <small className="form-hint">Leave empty to start immediately</small>
                        </div>

                        <div className="button-group">
                            <button 
                                className="primary-button" 
                                onClick={handleCommandSubmit}
                                disabled={isSending}
                            >
                                {isSending ? 'Sending...' : 'Send Command'}
                            </button>
                            <button 
                                className="secondary-button" 
                                onClick={handleCommandCancel}
                                disabled={isSending}
                            >
                                Cancel
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </>
    );
};

export default DeviceCard;