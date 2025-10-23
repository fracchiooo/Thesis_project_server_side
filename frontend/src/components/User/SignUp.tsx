import React, { useState, useEffect  } from 'react';
import axios from 'axios';
import '../../CSS/sign-up.css';


const UserSubscription: React.FC = () => {
  const [password, setPassword] = useState('');
  const [username, setUsername] = useState('');
  const [error, setError] = useState('');
  const [successMessage, setSuccessMessage] = useState('');
  const [isAuthorized, setIsAuthorized] = useState<boolean>(false);

  useEffect(() => {
    // Check if authToken exists in localStorage
    const token = localStorage.getItem('authToken');
    if (token) {
      setIsAuthorized(true);
    }
  }, []);

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();

    // Prepare the request body
    const body = {
      password,
      username,
    };

    try {
      // Send POST request to create the user profile
      const response = await axios.post('http://localhost:3000/user/create', body);
      
      // If successful, handle the success response
      setSuccessMessage(`Profile created successfully: ${JSON.stringify(response.data)}`);
      setError('');
      alert(`Profile ${username} created correctly!`);
      window.location.href = '/login';
    } catch (err: any) {
      // Handle errors (e.g., "Profile already created")
      if (err.response && err.response.status === 406) {
        setError(`Error: ${err.response.data}`);
      } else {
        setError('An unexpected error occurred');
      }
      setSuccessMessage('');
    }
  };

  if (isAuthorized) {
    return <p className="error-message">You need to be logged out to create a profile.</p>;
  }


  return (
    <div className="subscription-container">
      <h2 className="subscription-header">User Subscription</h2>
      <form onSubmit={handleSubmit} className="form-container">
      

        <div className="form-group">
          <label>Password:</label>
          <input 
            type="password" 
            value={password} 
            onChange={(e) => setPassword(e.target.value)} 
            required 
          />
        </div>

        <div className="form-group">
          <label>Username:</label>
          <input 
            type="text" 
            value={username} 
            onChange={(e) => setUsername(e.target.value)} 
            required 
          />
        </div>

        <button type="submit" className="create-button">Create Profile</button>
      </form>

      {error && <p className="error-message">{error}</p>}
      {successMessage && <p className="success-message">{successMessage}</p>}
        <div className="button-container">
          <button onClick={handleLogin} className="login-button">
              Login
          </button>
        </div>
    </div>
            
  );
};

const handleLogin = () => {

  window.location.href = '/login';

};

export default UserSubscription;