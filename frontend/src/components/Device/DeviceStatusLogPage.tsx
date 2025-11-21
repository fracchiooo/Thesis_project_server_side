import React, { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import axios from "axios";
import { DeviceStatusLog } from "../../types/DeviceStatusLog";
import { Line } from 'react-chartjs-2';
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend,
    TimeScale
} from 'chart.js';
import 'chartjs-adapter-date-fns';
import '../../CSS/device-status-log.css';

ChartJS.register(
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend,
    TimeScale
);

const formatDateForInput = (date: Date): string => {
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const day = String(date.getDate()).padStart(2, '0');
    const hours = String(date.getHours()).padStart(2, '0');
    const minutes = String(date.getMinutes()).padStart(2, '0');
    return `${year}-${month}-${day}T${hours}:${minutes}`;
};

const DeviceStatusLogPage = () => {
    const { deviceEUI } = useParams<{ deviceEUI: string }>();
    const navigate = useNavigate();
    const token = localStorage.getItem('authToken');

    const [allLogs, setAllLogs] = useState<DeviceStatusLog[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    const getDefaultDates = () => {
        const now = new Date();
        const thirtyDaysAgo = new Date();
        thirtyDaysAgo.setDate(now.getDate() - 30);
        
        return {
            start: formatDateForInput(thirtyDaysAgo),
            end: formatDateForInput(now)
        };
    };

    const [startDate, setStartDate] = useState(getDefaultDates().start);
    const [endDate, setEndDate] = useState(getDefaultDates().end);

    const parseInputDate = (dateString: string): Date => {
        return new Date(dateString + ':00');
    };

    const fetchAllLogs = async () => {
        if (!token || !deviceEUI) {
            navigate('/login');
            return;
        }

        const parsedData = JSON.parse(token);
        const tokenParsed = parsedData.token;

        try {
            setLoading(true);
            setError(null);

            const startDateObj = parseInputDate(startDate);
            const endDateObj = parseInputDate(endDate);

            let allFetchedLogs: DeviceStatusLog[] = [];
            let currentPage = 0;
            let totalPages = 1;
            const pageSize = 1000;

            while (currentPage < totalPages) {
                const response = await axios.get(
                    `http://localhost:3000/device/getDeviceLogsPages/${deviceEUI}`,
                    {
                        params: {
                            start_date: startDateObj.toISOString(),
                            finish_date: endDateObj.toISOString(),
                            page: currentPage,
                            size: pageSize
                        },
                        headers: {
                            Authorization: `Bearer ${tokenParsed}`
                        }
                    }
                );

                allFetchedLogs = [...allFetchedLogs, ...(response.data.content || [])];
                totalPages = response.data.totalPages || 1;
                currentPage++;

                console.log(`Fetched page ${currentPage} of ${totalPages}`);
            }

            setAllLogs(allFetchedLogs);
            console.log('Fetched all logs:', allFetchedLogs.length, 'total logs');
        } catch (err: any) {
            setError(err.message);
            console.error('Error fetching logs:', err);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchAllLogs();
    }, [startDate, endDate]);

    const temperatureChartData = {
        datasets: [
            {
                label: 'Temperature (°C)',
                data: allLogs.map(log => ({
                    x: log.statusDate ? new Date(log.statusDate) : null,
                    y: log.temperature?.valueOf() || null
                })).filter(d => d.x !== null && d.y !== null),
                borderColor: 'rgb(255, 99, 132)',
                backgroundColor: 'rgba(255, 99, 132, 0.5)',
                tension: 0.1,
                pointRadius: 2
            }
        ]
    };

    const frequencyChartData = {
        datasets: [
            {
                label: 'Frequency (kHz)',
                data: allLogs.map(log => ({
                    x: log.statusDate ? new Date(log.statusDate) : null,
                    y: log.frequency?.valueOf() ?? null
                })).filter(d => d.x !== null && d.y !== null),
                borderColor: 'rgb(54, 162, 235)',
                backgroundColor: 'rgba(54, 162, 235, 0.5)',
                tension: 0.1,
                pointRadius: 2
            }
        ]
    };

    const temperatureChartOptions = {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
            x: {
                type: 'time' as const,
                time: {
                    unit: 'hour' as const,
                    displayFormats: {
                        hour: 'MMM dd, HH:mm'
                    }
                },
                title: {
                    display: true,
                    text: 'Time'
                }
            },
            y: {
                beginAtZero: false,
                title: {
                    display: true,
                    text: 'Temperature (°C)'
                }
            }
        },
        plugins: {
            legend: {
                display: true,
                position: 'top' as const
            },
            tooltip: {
                mode: 'index' as const,
                intersect: false
            }
        }
    };

    const frequencyChartOptions = {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
            x: {
                type: 'time' as const,
                time: {
                    unit: 'hour' as const,
                    displayFormats: {
                        hour: 'MMM dd, HH:mm'
                    }
                },
                title: {
                    display: true,
                    text: 'Time'
                }
            },
            y: {
                beginAtZero: true,
                title: {
                    display: true,
                    text: 'Frequency (kHz)'
                }
            }
        },
        plugins: {
            legend: {
                display: true,
                position: 'top' as const
            },
            tooltip: {
                mode: 'index' as const,
                intersect: false
            }
        }
    };

    const handleDateChange = () => {
        fetchAllLogs();
    };

    if (loading) {
        return (
            <div className="loading">
                <div className="spinner"></div>
                <p>Loading device logs...</p>
            </div>
        );
    }

    if (error) {
        return <div className="error">Error: {error}</div>;
    }

    return (
        <div className="device-status-log-page">
            <div className="page-header">
                <h1>Device Logs: {deviceEUI}</h1>
                <button className="back-button" onClick={() => navigate('/devices')}>
                    ← Back to Devices
                </button>
            </div>

            <div className="date-range-selector">
                <div className="date-input-group">
                    <label>Start Date:</label>
                    <input
                        type="datetime-local"
                        value={startDate}
                        onChange={(e) => setStartDate(e.target.value)}
                        className="date-input"
                    />
                </div>

                <div className="date-input-group">
                    <label>End Date:</label>
                    <input
                        type="datetime-local"
                        value={endDate}
                        onChange={(e) => setEndDate(e.target.value)}
                        className="date-input"
                    />
                </div>

                <button className="apply-button" onClick={handleDateChange}>
                    Apply
                </button>
            </div>

            {allLogs.length > 0 ? (
                <>
                    <div className="logs-summary">
                        <p>Showing {allLogs.length} logs from {new Date(startDate).toLocaleString()} to {new Date(endDate).toLocaleString()}</p>
                    </div>

                    <div className="charts-container">
                        <div className="chart-wrapper">
                            <h2>Temperature Over Time</h2>
                            <div className="chart">
                                <Line data={temperatureChartData} options={temperatureChartOptions} />
                            </div>
                        </div>

                        <div className="chart-wrapper">
                            <h2>Frequency Over Time</h2>
                            <div className="chart">
                                <Line data={frequencyChartData} options={frequencyChartOptions} />
                            </div>
                        </div>
                    </div>
                </>
            ) : (
                <div className="no-data">
                    <p>No logs found for the selected date range.</p>
                    <p>Try selecting a different date range.</p>
                </div>
            )}
        </div>
    );
};

export default DeviceStatusLogPage;