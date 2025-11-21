import React, { useState, useEffect } from 'react';
import axios from 'axios';
import '../../CSS/login.css';


const Login: React.FC = () => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [isAuthorized, setIsAuthorized] = useState<boolean>(false);

  useEffect(() => {
    const token = localStorage.getItem('authToken');
    if (token) {
      setIsAuthorized(true);
    }
  }, []);

  const handleLogin = async () => {
    setError(null);

    try {
      const response = await axios.post('http://localhost:3000/user/login', {
        username,
        password,
      });

      const jwtToken = response.data;
      if (jwtToken && typeof jwtToken === 'string') {
        const tokenPayload = JSON.parse(
            atob(jwtToken.split('.')[1])
          );
      const expirationTime = tokenPayload.exp * 1000; 
      console.log("the expioration time is "+expirationTime);

    localStorage.setItem('authToken', JSON.stringify({ token: jwtToken, expiresAt: expirationTime }));
        alert('Login successful! with'+jwtToken);
        window.location.href = '/';
    } else {
        setError('Failed to retrieve token');
      }
    } catch (err) {
      console.error(err);
      setError('Login failed. Please check your username and password.');
    }
  };
  
  const handleSignUp = () => {

    window.location.href = '/signup';
  
  };

  if (isAuthorized) {
    return <p className="error-message">You need to be logged out to login.</p>;
  }

  return (
    <div className="login-container">
    <h2 className="login-header">Login</h2>
    <div className="form-group">
      <label>Username:</label>
      <input
        type="text"
        placeholder="Enter your username"
        value={username}
        onChange={(e) => setUsername(e.target.value)}
      />
    </div>
    <div className="form-group">
      <label>Password:</label>
      <input
        type="password"
        placeholder="Enter your password"
        value={password}
        onChange={(e) => setPassword(e.target.value)}
      />
    </div>
    <button onClick={handleLogin} className="login-button">
      Login
    </button>
    {error && <p className="error-message">{error}</p>}
    <div className="button-container">
      <button onClick={handleSignUp} className="signup-button">
        Signup
      </button>
    </div>
  </div>
  );
};

export default Login;