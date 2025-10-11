

import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { User, Lock, Eye, EyeOff } from 'lucide-react';

const GoaPoliceLogin = () => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const navigate = useNavigate();

  const API_BASE_URL = 'http://localhost:8000';

  const handleLogin = async (e) => {
    if (e) e.preventDefault();
    setIsLoading(true);
    setError('');

    try {
      const response = await fetch(`${API_BASE_URL}/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password }),
      });

      const data = await response.json();

      if (response.ok) {
        localStorage.setItem('access_token', data.access_token);
        localStorage.setItem('token_type', data.token_type);
        localStorage.setItem('username', username);
        navigate('/dashboard');
      } else {
        setError(data.detail || 'Login failed. Please check your credentials.');
      }
    } catch (error) {
      console.error('Login error:', error);
      setError('Connection error. Please check if the server is running.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') handleLogin();
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-white px-4 sm:px-6 lg:px-8">
      <div className="max-w-md w-full space-y-8 shadow-lg rounded-2xl p-8 bg-white border border-gray-100">
        
        {/* Logo + Header */}
        <div className="text-center">
          <div className="flex justify-center mb-4">
            <img 
              src="/police.png" 
              alt="Goa Police Logo" 
              className="h-32 w-auto object-contain"
            />
          </div>
          <h2 className="text-3xl font-extrabold text-gray-900">GOA POLICE</h2>
          <h3 className="text-lg font-medium text-gray-600 mt-2">Officer Login Portal</h3>
          <p className="text-sm text-gray-500 mt-1">Secure access to the management system</p>
        </div>

        {/* Error Message */}
        {error && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-4">
            <h3 className="text-sm font-semibold text-red-700">Authentication Error</h3>
            <p className="mt-1 text-sm text-red-600">{error}</p>
          </div>
        )}

        {/* Login Form */}
        <div className="space-y-4">
          {/* Username */}
          <div className="relative">
            <User className="absolute left-3 top-3 h-5 w-5 text-gray-400" />
            <input
              type="text"
              required
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Officer ID / Username"
              disabled={isLoading}
              className="w-full pl-10 pr-3 py-3 border rounded-lg shadow-sm text-gray-900 placeholder-gray-400 
                         focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 bg-white"
            />
          </div>

          {/* Password */}
          <div className="relative">
            <Lock className="absolute left-3 top-3 h-5 w-5 text-gray-400" />
            <input
              type={showPassword ? 'text' : 'password'}
              required
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Password"
              disabled={isLoading}
              className="w-full pl-10 pr-10 py-3 border rounded-lg shadow-sm text-gray-900 placeholder-gray-400 
                         focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 bg-white"
            />
            <button
              type="button"
              onClick={() => setShowPassword(!showPassword)}
              disabled={isLoading}
              className="absolute right-3 top-3 text-gray-400 hover:text-gray-600"
            >
              {showPassword ? <EyeOff className="h-5 w-5" /> : <Eye className="h-5 w-5" />}
            </button>
          </div>

          {/* Submit Button */}
          <button
            type="button"
            onClick={handleLogin}
            disabled={isLoading || !username.trim() || !password.trim()}
            className="w-full py-3 px-4 text-sm font-semibold rounded-lg shadow-md 
                       text-white bg-indigo-600 hover:bg-indigo-700 
                       focus:outline-none focus:ring-2 focus:ring-indigo-500 
                       disabled:opacity-50 disabled:cursor-not-allowed transition-all"
          >
            {isLoading ? (
              <div className="flex items-center justify-center">
                <div className="animate-spin h-4 w-4 border-b-2 border-white rounded-full mr-2"></div>
                Authenticating...
              </div>
            ) : (
              'Login Now'
            )}
          </button>
        </div>
      </div>
    </div>
  );
};

export default GoaPoliceLogin;

