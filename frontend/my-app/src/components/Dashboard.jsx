import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Shield, Mic, Waves, LogOut, User, UserPlus } from 'lucide-react';

const Dashboard = () => {
  const navigate = useNavigate();
  const username = localStorage.getItem('username') || 'Officer';

  const handleLogout = () => {
    localStorage.removeItem('access_token');
    localStorage.removeItem('token_type');
    localStorage.removeItem('username');
    navigate('/login');
  };

  const navigateToVoiceBiometric = () => {
    navigate('/voice-biometric');
  };

  const navigateToSyntheticDetection = () => {
    navigate('/synthetic-detection');
  };

  const navigateToInmateRegistration = () => {
    navigate('/voice-biometric');
  };

  const navigateToVoiceVerification = () => {
    navigate('/voice-biometric/verify');
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center space-x-3">
              <div className="bg-indigo-100 p-2 rounded-full">
                <Shield className="h-8 w-8 text-indigo-600" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-gray-900">GOA POLICE</h1>
                <p className="text-sm text-gray-600">Digital Forensics Platform</p>
              </div>
            </div>
            
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2 text-gray-700">
                <User className="h-5 w-5" />
                <span className="text-sm font-medium">Welcome, {username}</span>
              </div>
              <button
                onClick={handleLogout}
                className="flex items-center space-x-2 px-4 py-2 text-sm font-medium text-red-600 hover:text-red-700 hover:bg-red-50 rounded-lg transition-colors duration-200"
              >
                <LogOut className="h-4 w-4" />
                <span>Logout</span>
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {/* Page Title */}
        <div className="text-center mb-12">
          <h2 className="text-4xl font-bold text-gray-900 mb-4">
            Forensic Voice Analysis Dashboard
          </h2>
          <p className="text-lg text-gray-600 max-w-3xl mx-auto">
            Advanced voice analysis tools for law enforcement investigations. 
            Select the appropriate analysis method for your case.
          </p>
        </div>

        {/* Dashboard Cards */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 max-w-7xl mx-auto">
          {/* Voice Biometric Card */}
          <div className="group relative bg-white rounded-2xl shadow-lg hover:shadow-2xl transition-all duration-300 border border-gray-200 overflow-hidden">
            {/* Gradient Background */}
            <div className="absolute inset-0 bg-gradient-to-br from-indigo-50 to-purple-50 opacity-50 group-hover:opacity-70 transition-opacity duration-300"></div>
            
            {/* Card Content */}
            <div className="relative p-6">
              <div className="flex items-center justify-center w-14 h-14 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-2xl mb-5 group-hover:scale-110 transition-transform duration-300">
                <Mic className="h-7 w-7 text-white" />
              </div>
              
              <h3 className="text-xl font-bold text-gray-900 mb-3 group-hover:text-indigo-600 transition-colors duration-300">
                Voice Analysis
              </h3>
              
              <p className="text-gray-600 mb-5 leading-relaxed text-sm">
                Speaker verification and identification using advanced voice biometric technology. 
                Compare voice samples against known suspects.
              </p>
              
              <ul className="space-y-2 mb-6">
                <li className="flex items-center text-xs text-gray-600">
                  <div className="w-2 h-2 bg-indigo-500 rounded-full mr-3"></div>
                  Voice Verification
                </li>
                <li className="flex items-center text-xs text-gray-600">
                  <div className="w-2 h-2 bg-indigo-500 rounded-full mr-3"></div>
                  Speaker Identification
                </li>
              </ul>
              
              {/* Action Buttons */}
              <div className="space-y-2">
                <button
                  onClick={navigateToVoiceVerification}
                  className="w-full flex items-center justify-center px-4 py-2 bg-indigo-600 hover:bg-indigo-700 text-white text-sm font-medium rounded-lg transition-colors duration-200"
                >
                  Verify Voice Sample
                </button>
                
                <button
                  onClick={navigateToVoiceBiometric}
                  className="w-full flex items-center justify-center px-4 py-2 border border-indigo-600 text-indigo-600 hover:bg-indigo-50 text-sm font-medium rounded-lg transition-colors duration-200"
                >
                  Analyze Audio
                </button>
              </div>
            </div>
          </div>

          {/* Inmate Registration Card */}
          <div
            onClick={navigateToInmateRegistration}
            className="group relative bg-white rounded-2xl shadow-lg hover:shadow-2xl transition-all duration-300 cursor-pointer transform hover:scale-105 border border-gray-200 overflow-hidden"
          >
            {/* Gradient Background */}
            <div className="absolute inset-0 bg-gradient-to-br from-green-50 to-teal-50 opacity-50 group-hover:opacity-70 transition-opacity duration-300"></div>
            
            {/* Card Content */}
            <div className="relative p-6">
              <div className="flex items-center justify-center w-14 h-14 bg-gradient-to-br from-green-500 to-teal-600 rounded-2xl mb-5 group-hover:scale-110 transition-transform duration-300">
                <UserPlus className="h-7 w-7 text-white" />
              </div>
              
              <h3 className="text-xl font-bold text-gray-900 mb-3 group-hover:text-green-600 transition-colors duration-300">
                Register Inmate
              </h3>
              
              <p className="text-gray-600 mb-5 leading-relaxed text-sm">
                Register new inmate profiles with voice biometric data. Create reference 
                voice prints for future identification and verification tasks.
              </p>
              
              <ul className="space-y-2 mb-6">
                <li className="flex items-center text-xs text-gray-600">
                  <div className="w-2 h-2 bg-green-500 rounded-full mr-3"></div>
                  Voice Profile Creation
                </li>
                <li className="flex items-center text-xs text-gray-600">
                  <div className="w-2 h-2 bg-green-500 rounded-full mr-3"></div>
                  Biometric Registration
                </li>
              </ul>
              
              <div className="flex items-center text-green-600 font-medium group-hover:text-green-700 text-sm">
                <span>Register New</span>
                <svg className="w-3 h-3 ml-2 group-hover:translate-x-1 transition-transform duration-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5l7 7-7 7"></path>
                </svg>
              </div>
            </div>
          </div>

          {/* Synthetic Voice Detection Card */}
          <div
            onClick={navigateToSyntheticDetection}
            className="group relative bg-white rounded-2xl shadow-lg hover:shadow-2xl transition-all duration-300 cursor-pointer transform hover:scale-105 border border-gray-200 overflow-hidden"
          >
            {/* Gradient Background */}
            <div className="absolute inset-0 bg-gradient-to-br from-purple-50 to-pink-50 opacity-50 group-hover:opacity-70 transition-opacity duration-300"></div>
            
            {/* Card Content */}
            <div className="relative p-6">
              <div className="flex items-center justify-center w-14 h-14 bg-gradient-to-br from-purple-500 to-pink-600 rounded-2xl mb-5 group-hover:scale-110 transition-transform duration-300">
                <Waves className="h-7 w-7 text-white" />
              </div>
              
              <h3 className="text-xl font-bold text-gray-900 mb-3 group-hover:text-purple-600 transition-colors duration-300">
                Synthetic Detection
              </h3>
              
              <p className="text-gray-600 mb-5 leading-relaxed text-sm">
                Detect artificially generated speech, deepfake audio, and synthetic voice 
                recordings using state-of-the-art machine learning algorithms.
              </p>
              
              <ul className="space-y-2 mb-6">
                <li className="flex items-center text-xs text-gray-600">
                  <div className="w-2 h-2 bg-purple-500 rounded-full mr-3"></div>
                  Deepfake Detection
                </li>
                <li className="flex items-center text-xs text-gray-600">
                  <div className="w-2 h-2 bg-purple-500 rounded-full mr-3"></div>
                  AI Voice Analysis
                </li>
              </ul>
              
              <div className="flex items-center text-purple-600 font-medium group-hover:text-purple-700 text-sm">
                <span>Start Detection</span>
                <svg className="w-3 h-3 ml-2 group-hover:translate-x-1 transition-transform duration-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5l7 7-7 7"></path>
                </svg>
              </div>
            </div>
          </div>
        </div>

        {/* Additional Information */}
        <div className="mt-16 text-center">
          <div className="bg-white rounded-xl shadow-lg p-8 max-w-4xl mx-auto border border-gray-200">
            <div className="flex justify-center mb-4">
              <div className="bg-gradient-to-r from-indigo-100 to-purple-100 p-3 rounded-full">
                <Shield className="h-8 w-8 text-indigo-600" />
              </div>
            </div>
            <h3 className="text-xl font-bold text-gray-900 mb-4">
              Forensic-Grade Voice Analysis
            </h3>
            <p className="text-gray-600 leading-relaxed">
              All analysis results are generated using forensic-grade algorithms and include 
              detailed reports with confidence scores, technical metadata, and digital signatures 
              for legal admissibility in court proceedings.
            </p>
            <div className="mt-6 flex justify-center space-x-8 text-sm text-gray-500">
              <div className="flex items-center">
                <div className="w-2 h-2 bg-green-500 rounded-full mr-2"></div>
                Court Admissible
              </div>
              <div className="flex items-center">
                <div className="w-2 h-2 bg-green-500 rounded-full mr-2"></div>
                Digitally Signed
              </div>
              <div className="flex items-center">
                <div className="w-2 h-2 bg-green-500 rounded-full mr-2"></div>
                Secure Processing
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
};

export default Dashboard;