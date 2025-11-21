import React, { useEffect, useState, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";
import DeviceCard from "./DeviceCard.tsx";
import { FaPlus, FaSync } from "react-icons/fa";
import { Device } from "../../types/Device";
import '../../CSS/device.css';

interface StatusDto {
    deviceEUI: string;
    lastUpdate: Date;
    currentTemperature: number;
    currentSensedFrequency: number;
    username: string;
    deviceEnvRequests: Record<string, any>;
}

const DevicePage = () => {
    const [devices, setDevices] = useState<Device[]>([]);
    const [statuses, setStatuses] = useState<StatusDto[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [lastRefresh, setLastRefresh] = useState<Date>(new Date());
    const [autoRefresh, setAutoRefresh] = useState(true);
    const token = localStorage.getItem('authToken');
    const navigate = useNavigate();
    const [isModalOpen, setIsModalOpen] = useState(false);
    const [newDeviceName, setNewDeviceName] = useState('');

    const fetchDevices = useCallback(async () => {
        if (token == null) {
            navigate('/login');
            return;
        }
        const parsedData = JSON.parse(token);
        const tokenParsed = parsedData.token;
        
        try {
            setLoading(true);
            setError(null);
            const response = await axios.get<Device[]>("http://localhost:3000/device/list", {
                headers: {
                    Authorization: `Bearer ${tokenParsed}`
                }
            });
            const data = response.data;
            setDevices(data);
            console.log("Fetched devices:", data);
        } catch (err: any) {
            setError(err.message);
            console.error("Error fetching devices:", err);
        } finally {
            setLoading(false);
        }
    }, [token, navigate]);

    const fetchStatuses = useCallback(async () => {
        if (token == null) {
            return;
        }
        const parsedData = JSON.parse(token);
        const tokenParsed = parsedData.token;
        
        try {
            const response = await axios.get<StatusDto[]>("http://localhost:3000/device/getAllStatusses", {
                headers: {
                    Authorization: `Bearer ${tokenParsed}`
                }
            });
            const data = response.data;
            setStatuses(data);
            setLastRefresh(new Date());
            console.log("Fetched statuses:", data);

            setDevices(prevDevices => 
                prevDevices.map(device => {
                    const cleanDeviceEUI = device.deviceEUI?.replace(/^"|"$/g, '');
                    const status = data.find(s => s.deviceEUI === cleanDeviceEUI);
                    if (status) {
                        return {
                            ...device,
                            lastUpdate: status.lastUpdate,
                            currentTemperature: status.currentTemperature,
                            currentSensedFrequency: status.currentSensedFrequency,
                            deviceEnvRequests: status.deviceEnvRequests
                        };
                    }
                    return device;
                })
            );
        } catch (err: any) {
            console.error("Error fetching statuses:", err);
        }
    }, [token]);

    useEffect(() => {
        fetchDevices();
    }, [fetchDevices]);

    useEffect(() => {
        if (!autoRefresh) return;

        const interval = setInterval(() => {
            fetchStatuses();
        }, 30000);

        fetchStatuses();

        return () => clearInterval(interval);
    }, [autoRefresh, fetchStatuses]);

    const handleManualRefresh = () => {
        fetchStatuses();
    };

    const handleDelete = async (deviceEUI: string) => {
        if (token == null) throw new Error("Token is null");
        
        const cleanEUI = deviceEUI.replace(/^"|"$/g, '');
        
        const parsedData = JSON.parse(token);
        const tokenParsed = parsedData.token;
        
        if (!window.confirm(`Are you sure you want to delete device ${cleanEUI}?`)) {
            return;
        }

        try {
            setLoading(true);
            await axios.post(`http://localhost:3000/device/delete`, { devEUI: cleanEUI }, {
                headers: {
                    Authorization: `Bearer ${tokenParsed}`
                }
            });
            alert('Device deleted successfully!');
        } catch (err: any) {
            setError(err.message);
            alert('Failed to delete device: ' + err.message);
        } finally {
            setLoading(false);
            fetchDevices();
        }
    };

    const onCreate = () => {
        setIsModalOpen(true);
    };

    const handleCreate = async () => {
        if (token == null) throw new Error("Token is null");
        if (!newDeviceName.trim()) {
            alert('Please enter a device EUI');
            return;
        }

        const parsedData = JSON.parse(token);
        const tokenParsed = parsedData.token;

        try {
            setLoading(true);
            await axios.post(
                "http://localhost:3000/device/create", 
                newDeviceName,
                {
                    headers: {
                        Authorization: `Bearer ${tokenParsed}`,
                        'Content-Type': 'text/plain'
                    }
                }
            );
            alert('Device created successfully!');
            setIsModalOpen(false);
            setNewDeviceName('');
        } catch (err: any) {
            setError(err.message);
            alert('Failed to create device: ' + err.message);
        } finally {
            setLoading(false);
            fetchDevices();
        }
    };

    const handleCancel = () => {
        setIsModalOpen(false);
        setNewDeviceName('');
    };

    const formatLastRefresh = () => {
        const now = new Date();
        const diff = Math.floor((now.getTime() - lastRefresh.getTime()) / 1000);
        
        if (diff < 60) return `${diff} seconds ago`;
        if (diff < 3600) return `${Math.floor(diff / 60)} minutes ago`;
        return lastRefresh.toLocaleTimeString();
    };

    if (loading && devices.length === 0) return <div className="loading">Loading devices...</div>;
    if (error) return <div className="error">Error: {error}</div>;

    return (
        <div className="device-page-wrapper">
            <div className="page-header">
                <h1>My Devices</h1>
                <div className="header-actions">
                    <div className="refresh-controls">
                        <button 
                            className="refresh-button" 
                            onClick={handleManualRefresh}
                            title="Refresh now"
                        >
                            <FaSync /> Refresh
                        </button>
                        <label className="auto-refresh-toggle">
                            <input
                                type="checkbox"
                                checked={autoRefresh}
                                onChange={(e) => setAutoRefresh(e.target.checked)}
                            />
                            <span>Auto-refresh (30s)</span>
                        </label>
                        <span className="last-refresh">Last: {formatLastRefresh()}</span>
                    </div>
                    <button className="create-button" onClick={onCreate}>
                        <FaPlus /> Create New Device
                    </button>
                </div>
            </div>
            
            <div className="device-container">
                {devices.length > 0 ? (
                    devices.map((device, index) => (
                        <DeviceCard 
                            key={device.deviceEUI || index} 
                            device={device}
                            onDelete={handleDelete} 
                        />
                    ))
                ) : (
                    <div className="no-devices">
                        <p>No devices found.</p>
                        <p>Click "Create New Device" to add your first device.</p>
                    </div>
                )}
            </div>

            {isModalOpen && (
                <div className="modal-overlay" onClick={handleCancel}>
                    <div className="modal-content" onClick={(e) => e.stopPropagation()}>
                        <h2>Create New Device</h2>
                        <div className="form-group">
                            <label>Device EUI:</label>
                            <input
                                type="text"
                                value={newDeviceName}
                                onChange={(e) => setNewDeviceName(e.target.value)}
                                placeholder="Enter device EUI (e.g., 10:20:ba:66:2e:94)"
                            />
                        </div>
                        <div className="button-group">
                            <button className="primary-button" onClick={handleCreate}>
                                Create
                            </button>
                            <button className="secondary-button" onClick={handleCancel}>
                                Cancel
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default DevicePage;