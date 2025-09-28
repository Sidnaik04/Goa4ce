import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import GoaPoliceLogin from './components/GoaPoliceLogin';
import Dashboard from './components/Dashboard';
import VoiceBiometricPage from './components/VoiceBiometricPage';
import VoiceBiometricResults from './components/VoiceBiometricResults';
import VoiceVerificationPage from './components/VoiceVerificationPage';
import SyntheticDetectionPage from './components/SyntheticDetectionPage';
import ProtectedRoute from './components/ProtectedRoute';

function App() {
  // Check if user is already logged in
  const isAuthenticated = () => {
    const token = localStorage.getItem('access_token');
    if (!token) return false;

    try {
      const tokenPayload = JSON.parse(atob(token.split('.')[1]));
      const currentTime = Date.now() / 1000;
      return tokenPayload.exp > currentTime;
    } catch (error) {
      return false;
    }
  };

  return (
    <Router>
      <div className="App">
        <Routes>
          {/* Public Route - Login */}
          <Route 
            path="/login" 
            element={
              isAuthenticated() ? 
              <Navigate to="/dashboard" replace /> : 
              <GoaPoliceLogin />
            } 
          />
          
          {/* Protected Routes */}
          <Route 
            path="/dashboard" 
            element={
              <ProtectedRoute>
                <Dashboard />
              </ProtectedRoute>
            } 
          />
          
          <Route 
            path="/voice-biometric" 
            element={
              <ProtectedRoute>
                <VoiceBiometricPage />
              </ProtectedRoute>
            } 
          />
          
          <Route 
            path="/voice-biometric/results" 
            element={
              <ProtectedRoute>
                <VoiceBiometricResults />
              </ProtectedRoute>
            } 
          />
          
          <Route 
            path="/voice-biometric/verify" 
            element={
              <ProtectedRoute>
                <VoiceVerificationPage />
              </ProtectedRoute>
            } 
          />
          
          <Route 
            path="/synthetic-detection" 
            element={
              <ProtectedRoute>
                <SyntheticDetectionPage />
              </ProtectedRoute>
            } 
          />
          
          {/* Default redirect */}
          <Route 
            path="/" 
            element={
              <Navigate to={isAuthenticated() ? "/dashboard" : "/login"} replace />
            } 
          />
          
          {/* Catch all - redirect to appropriate page */}
          <Route 
            path="*" 
            element={
              <Navigate to={isAuthenticated() ? "/dashboard" : "/login"} replace />
            } 
          />
        </Routes>
      </div>
    </Router>
  );
}

export default App;