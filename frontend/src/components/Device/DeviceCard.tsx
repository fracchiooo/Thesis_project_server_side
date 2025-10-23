import React from "react";
import { FaTrash, FaThermometerHalf, FaClock, FaBroadcastTower } from "react-icons/fa";
import { useNavigate } from "react-router-dom";
import { Device } from "../../types/Device";
import '../../CSS/device.css'; 

interface DeviceCardProps {
    device: Device;
    onDelete: (deviceEUI: string) => void;
}

const DeviceCard: React.FC<DeviceCardProps> = ({ device, onDelete }) => {
    const navigate = useNavigate();
    
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
        // Naviga alla pagina di dettaglio del device (se esiste)
        // navigate(`/device/${cleanDeviceEUI}`);
    };

    return (
        <div className="device-card">
            <div className="device-card-header">
                <span className="device-eui">
                    {cleanDeviceEUI}
                </span>
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
    );
};

export default DeviceCard;