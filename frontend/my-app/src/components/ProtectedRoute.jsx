import React, { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';

const ProtectedRoute = ({ children }) => {
  const navigate = useNavigate();

  useEffect(() => {
    const token = localStorage.getItem('access_token');
    
    if (!token) {
      // No token found, redirect to login
      navigate('/login', { replace: true });
      return;
    }

    // Optional: Verify token is not expired
    try {
      // Decode JWT token to check expiration (optional)
      const tokenPayload = JSON.parse(atob(token.split('.')[1]));
      const currentTime = Date.now() / 1000;
      
      if (tokenPayload.exp < currentTime) {
        // Token is expired
        localStorage.removeItem('access_token');
        localStorage.removeItem('token_type');
        localStorage.removeItem('username');
        navigate('/login', { replace: true });
        return;
      }
    } catch (error) {
      // Invalid token format
      console.error('Invalid token format:', error);
      localStorage.removeItem('access_token');
      localStorage.removeItem('token_type');
      localStorage.removeItem('username');
      navigate('/login', { replace: true });
      return;
    }
  }, [navigate]);

  const token = localStorage.getItem('access_token');
  
  // Don't render children if no token
  if (!token) {
    return null;
  }

  return children;
};

export default ProtectedRoute;