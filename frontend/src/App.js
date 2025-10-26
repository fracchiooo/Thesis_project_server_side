import './App.css';
import Login from './components/User/Login.tsx';
import UserSubscription from './components/User/SignUp.tsx';
import PredictionPage from './components/Prediction/PredictionPage.tsx';
import DevicePage from './components/Device/DevicePage.tsx';
import DeviceStatusLogPage from './components/Device/DeviceStatusLogPage.tsx';

import { useNavigate } from "react-router-dom";
import { useEffect } from "react";
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';

function App() {
  useEffect(() => {
    document.body.style.backgroundColor = '#88c265';
  }, []);

  return (
    <Router>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/signup" element={<UserSubscription />} />
        <Route path="/login" element={<Login />} />
        <Route path="/predictions" element={<PredictionPage/>} />
        <Route path="/devices" element={<DevicePage/>} />
        <Route path="*" element={<h1>Page not found</h1>} />
        <Route path="/device/:deviceEUI" element={<DeviceStatusLogPage />} />
      </Routes>
    </Router>
  );
}

// HomePage è DENTRO il Router, quindi può usare useNavigate
function HomePage() {
  const isLoggedIn = localStorage.getItem('authToken') != null;
  const navigate = useNavigate(); // Ora funziona!

  const handleSignUp = () => {
    navigate('/signup');
  };

  const handleLogin = () => {
    navigate('/login');
  };
    
  
  const handlePredictions = () => {
    navigate('/predictions');
  };

  const handleDevices = () => {
    navigate('/devices');
  };

  const handleLogout = () => {
    localStorage.removeItem('authToken');
    alert('Logged out successfully!');
    window.location.reload();
  };

  return (
    <div className="app-container">
      <h1 className="app-title"></h1>
      
      {isLoggedIn ? (
        <div className="button-container">
          <button className="logout-button" onClick={handleLogout}>Logout</button>
          <button className="devices-button" onClick={handleDevices}>My Devices</button>
          <button className="predictions-button" onClick={handlePredictions}>My Predictions</button>

        </div>
      ) : (
        <div className="button-container">
          <button className="nav-button" onClick={handleLogin}>Login</button>
          <button className="nav-button" onClick={handleSignUp}>Sign Up</button>
        </div>
      )}
    </div>
  );
}

export default App;