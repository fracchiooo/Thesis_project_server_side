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
        finish_after_hours: '',
        finish_after_minutes: '',
        finish_after_seconds: '',
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

    // Converti ore, minuti, secondi in ore decimali
    const convertToDecimalHours = (hours: string, minutes: string, seconds: string): number => {
        const h = parseFloat(hours) || 0;
        const m = parseFloat(minutes) || 0;
        const s = parseFloat(seconds) || 0;
        
        return h + (m / 60.0) + (s / 3600.0);
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
        if (!commandData.frequency || !commandData.duty_frequency) {
            errors.push('Frequency and Duty Frequency are required');
            setValidationErrors(errors);
            return false;
        }

        // Verifica che almeno uno dei campi tempo sia compilato
        const hasTime = commandData.finish_after_hours || 
                       commandData.finish_after_minutes || 
                       commandData.finish_after_seconds;
        
        if (!hasTime) {
            errors.push('At least one time field (hours, minutes, or seconds) must be filled');
        }

        const frequency = parseFloat(commandData.frequency);
        const dutyFrequency = parseFloat(commandData.duty_frequency);

        // Validazione range frequency (0-40 kHz)
        if (isNaN(frequency) || frequency < 0 || frequency > 40) {
            errors.push('Frequency must be between 0 and 40 kHz');
        }

        // Validazione range duty_frequency (20-100)
        if (isNaN(dutyFrequency) || dutyFrequency < 20 || dutyFrequency > 100) {
            errors.push('Duty Frequency must be between 20 and 100');
        }

        // Validazione finish_after (maggiore di 0)
        const finishAfterHours = convertToDecimalHours(
            commandData.finish_after_hours,
            commandData.finish_after_minutes,
            commandData.finish_after_seconds
        );

        if (finishAfterHours <= 0) {
            errors.push('Total duration must be greater than 0');
        }

        // Validazione valori negativi per tempo
        const hours = parseFloat(commandData.finish_after_hours) || 0;
        const minutes = parseFloat(commandData.finish_after_minutes) || 0;
        const seconds = parseFloat(commandData.finish_after_seconds) || 0;

        if (hours < 0 || minutes < 0 || seconds < 0) {
            errors.push('Time values cannot be negative');
        }

        if (minutes >= 60) {
            errors.push('Minutes must be less than 60');
        }

        if (seconds >= 60) {
            errors.push('Seconds must be less than 60');
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

            // Converti il tempo in ore decimali
            const finishAfterHours = convertToDecimalHours(
                commandData.finish_after_hours,
                commandData.finish_after_minutes,
                commandData.finish_after_seconds
            );

            const commandDto: any = {
                frequency: parseFloat(commandData.frequency),
                duty_frequency: parseFloat(commandData.duty_frequency),
                finish_after: finishAfterHours
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
                finish_after_hours: '',
                finish_after_minutes: '',
                finish_after_seconds: '',
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
            finish_after_hours: '',
            finish_after_minutes: '',
            finish_after_seconds: '',
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
                                ? `${device.currentSensedFrequency} kHz` 
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
                                Frequency (kHz): <span className="required">*</span>
                                <span className="range-hint">(0 - 40)</span>
                            </label>
                            <input
                                type="number"
                                step="0.01"
                                min="0"
                                max="40"
                                value={commandData.frequency}
                                onChange={(e) => setCommandData(prev => ({ ...prev, frequency: e.target.value }))}
                                placeholder="Enter frequency (0-40 kHz)"
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
                                min="20"
                                max="100"
                                value={commandData.duty_frequency}
                                onChange={(e) => setCommandData(prev => ({ ...prev, duty_frequency: e.target.value }))}
                                placeholder="Enter duty frequency (20-100)"
                                required
                            />
                        </div>

                        <div className="form-group">
                            <label>
                                Duration: <span className="required">*</span>
                            </label>
                            <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
                                <div style={{ flex: 1 }}>
                                    <input
                                        type="number"
                                        step="1"
                                        min="0"
                                        value={commandData.finish_after_hours}
                                        onChange={(e) => setCommandData(prev => ({ 
                                            ...prev, 
                                            finish_after_hours: e.target.value 
                                        }))}
                                        placeholder="Hours"
                                    />
                                    <small>hours</small>
                                </div>
                                <div style={{ flex: 1 }}>
                                    <input
                                        type="number"
                                        step="1"
                                        min="0"
                                        max="59"
                                        value={commandData.finish_after_minutes}
                                        onChange={(e) => setCommandData(prev => ({ 
                                            ...prev, 
                                            finish_after_minutes: e.target.value 
                                        }))}
                                        placeholder="Minutes"
                                    />
                                    <small>minutes</small>
                                </div>
                                <div style={{ flex: 1 }}>
                                    <input
                                        type="number"
                                        step="1"
                                        min="0"
                                        max="59"
                                        value={commandData.finish_after_seconds}
                                        onChange={(e) => setCommandData(prev => ({ 
                                            ...prev, 
                                            finish_after_seconds: e.target.value 
                                        }))}
                                        placeholder="Seconds"
                                    />
                                    <small>seconds</small>
                                </div>
                            </div>
                            <small className="form-hint">Fill at least one field</small>
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