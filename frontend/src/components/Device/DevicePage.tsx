import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";
import DeviceCard from "./DeviceCard.tsx";
import { FaPlus } from "react-icons/fa";
import { Device } from "../../types/Device";
import '../../CSS/device.css';

const DevicePage = () => {
    const [devices, setDevices] = useState<Device[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const token = localStorage.getItem('authToken');
    const navigate = useNavigate();
    const [isModalOpen, setIsModalOpen] = useState(false);
    const [newDeviceName, setNewDeviceName] = useState('');

    const fetchDevices = async () => {
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
    };

    useEffect(() => {
        fetchDevices();
    }, []);

    const handleDelete = async (deviceEUI: string) => {
        if (token == null) throw new Error("Token is null");
        
        // Rimuovi virgolette extra se presenti
        const cleanEUI = deviceEUI.replace(/^"|"$/g, '');
        
        const parsedData = JSON.parse(token);
        const tokenParsed = parsedData.token;
        
        if (!window.confirm(`Are you sure you want to delete device ${cleanEUI}?`)) {
            return;
        }

        try {
            setLoading(true);
            await axios.post(`http://localhost:3000/device/delete`,{devEUI : cleanEUI}, {
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

    if (loading) return <div className="loading">Loading devices...</div>;
    if (error) return <div className="error">Error: {error}</div>;

    return (
        <div className="device-wrapper">
            <div className="page-header">
                <h1>My Devices</h1>
                <button className="create-button" onClick={onCreate}>
                    <FaPlus /> Create New Device
                </button>
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