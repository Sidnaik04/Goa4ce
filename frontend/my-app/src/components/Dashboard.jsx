
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

  const navigateToVoiceBiometric = () => navigate('/voice-biometric');
  const navigateToSyntheticDetection = () => navigate('/synthetic-detection');
  const navigateToInmateRegistration = () => navigate('/voice-biometric');
  const navigateToVoiceVerification = () => navigate('/voice-biometric/verify');

  return (
    <div className="min-h-screen bg-white flex flex-col">
      {/* Navbar */}
      <header className="fixed top-0 left-0 right-0 bg-white shadow-md z-50 border-b">
        <div className="max-w-7xl mx-auto px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            {/* Brand */}
            <div className="flex items-center space-x-3">
              <img 
                src="/police.png" 
                alt="Goa Police Logo" 
                className="h-10 w-auto object-contain"
              />
              <div>
                <h1 className="text-lg font-bold text-gray-900">GOA POLICE</h1>
                <p className="text-xs text-gray-500">Digital Forensics Platform</p>
              </div>
            </div>

            {/* User + Logout */}
            <div className="flex items-center space-x-5">
              <div className="flex items-center text-gray-900 space-x-2 bg-blue-50 px-4 py-2 rounded-lg border border-blue-200 shadow-sm">
  <User className="h-5 w-5 text-blue-600" />
  <span className="text-sm font-semibold">Welcome, <span className="text-blue-600">{username}</span></span>
</div>
              <button
                onClick={handleLogout}
                className="flex items-center space-x-2 px-4 py-2 text-sm font-medium text-white bg-red-600 hover:bg-red-700 rounded-lg transition-colors shadow-sm"
              >
                <LogOut className="h-4 w-4" />
                <span>Logout</span>
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-28 pb-16">
        {/* Page Title */}
        <div className="text-center mb-14">
          <h2 className="text-4xl font-extrabold text-gray-900 mb-4">
            Forensic Voice Analysis Dashboard
          </h2>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            Advanced voice analysis tools for law enforcement investigations. 
            Select the appropriate analysis method for your case.
          </p>
        </div>

        {/* Dashboard Cards */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Voice Biometric Card */}
          <div className="group bg-white rounded-xl shadow-lg hover:shadow-2xl transition-all duration-300 border-2 border-gray-300 overflow-hidden">
            <div className="p-6 relative">
              <div className="flex items-center justify-center mb-5">
  <img 
    src="/biometric.png" 
    alt="Voice Analysis" 
    className="h-16 w-auto object-contain" 
  />
</div>
              <h3 className="text-lg font-bold text-gray-900 mb-2 group-hover:text-indigo-600">Voice Analysis</h3>
              <p className="text-gray-600 text-sm mb-5">
                Speaker verification and identification using advanced voice biometric technology.
              </p>
              <ul className="space-y-2 mb-6 text-xs text-gray-600">
                <li className="flex items-center"><div className="w-2 h-2 bg-indigo-500 rounded-full mr-2"></div>Voice Verification</li>
                <li className="flex items-center"><div className="w-2 h-2 bg-indigo-500 rounded-full mr-2"></div>Speaker Identification</li>
              </ul>
              <div className="space-y-2">
                <button
                  onClick={navigateToVoiceVerification}
                  className="w-full py-2 bg-indigo-600 hover:bg-indigo-700 text-white text-sm font-medium rounded-lg transition"
                >
                  Verify Voice Sample
                </button>
                {/* <button
                  onClick={navigateToVoiceBiometric}
                  className="w-full py-2 border border-indigo-600 text-indigo-600 hover:bg-indigo-50 text-sm font-medium rounded-lg transition"
                >
                  Analyze Audio
                </button> */}
              </div>
            </div>
          </div>

          {/* Inmate Registration Card */}
          <div
            onClick={navigateToInmateRegistration}
            className="group bg-white rounded-xl shadow-lg hover:shadow-2xl transition-all duration-300 border-2 border-gray-300 cursor-pointer"
          >
            <div className="p-6">
              <div className="flex items-center justify-center mb-5">
  <img 
    src="/inmate.png" 
    alt="Register Inmate" 
    className="h-16 w-auto object-contain" 
  />
</div>
              <h3 className="text-lg font-bold text-gray-900 mb-2 group-hover:text-green-600">Register Inmate</h3>
              <p className="text-gray-600 text-sm mb-5">
                Register new inmate profiles with voice biometric data. Create reference voice prints for identification.
              </p>
              <ul className="space-y-2 mb-6 text-xs text-gray-600">
                <li className="flex items-center"><div className="w-2 h-2 bg-green-500 rounded-full mr-2"></div>Voice Profile Creation</li>
                <li className="flex items-center"><div className="w-2 h-2 bg-green-500 rounded-full mr-2"></div>Biometric Registration</li>
              </ul>
              <div className="flex items-center text-green-600 font-medium text-sm group-hover:text-green-700">
                Register New
                <svg className="w-3 h-3 ml-2 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5l7 7-7 7"></path>
                </svg>
              </div>
            </div>
          </div>

          {/* Synthetic Voice Detection Card */}
          <div
            onClick={navigateToSyntheticDetection}
            className="group bg-white rounded-xl shadow-lg hover:shadow-2xl transition-all duration-300 border-2 border-gray-300 cursor-pointer"
          >
            <div className="p-6">
              <div className="flex items-center justify-center mb-5">
  <img 
    src="/synthetic.png" 
    alt="Synthetic Detection" 
    className="h-16 w-auto object-contain" 
  />
</div>
              <h3 className="text-lg font-bold text-gray-900 mb-2 group-hover:text-purple-600">Synthetic Detection</h3>
              <p className="text-gray-600 text-sm mb-5">
                Detect artificially generated speech, deepfake audio, and synthetic recordings using AI.
              </p>
              <ul className="space-y-2 mb-6 text-xs text-gray-600">
                <li className="flex items-center"><div className="w-2 h-2 bg-purple-500 rounded-full mr-2"></div>Deepfake Detection</li>
                <li className="flex items-center"><div className="w-2 h-2 bg-purple-500 rounded-full mr-2"></div>AI Voice Analysis</li>
              </ul>
              <div className="flex items-center text-purple-600 font-medium text-sm group-hover:text-purple-700">
                Start Detection
                <svg className="w-3 h-3 ml-2 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5l7 7-7 7"></path>
                </svg>
              </div>
            </div>
          </div>
        </div>

        {/* Info Box */}
        <div className="mt-20 text-center">
          <div className="bg-white rounded-xl shadow-lg p-8 max-w-4xl mx-auto border-2 border-gray-300">
            <div className="flex justify-center mb-4">
  <img 
    src="/voice.jpg" 
    alt="Forensic Voice Analysis" 
    className="h-32 w-auto object-contain" 
  />
</div>
            <h3 className="text-xl font-bold text-gray-900 mb-4">Forensic-Grade Voice Analysis</h3>
            <p className="text-gray-600 leading-relaxed text-sm">
              All analysis results use forensic-grade algorithms with detailed reports, confidence scores, technical metadata, 
              and digital signatures — ensuring legal admissibility in court proceedings.
            </p>
            <div className="mt-6 flex justify-center space-x-8 text-xs text-gray-500">
              <div className="flex items-center"><div className="w-2 h-2 bg-green-500 rounded-full mr-2"></div>Court Admissible</div>
              <div className="flex items-center"><div className="w-2 h-2 bg-green-500 rounded-full mr-2"></div>Digitally Signed</div>
              <div className="flex items-center"><div className="w-2 h-2 bg-green-500 rounded-full mr-2"></div>Secure Processing</div>
            </div>
          </div>
        </div>
      </main>

      {/* Footer */}
<footer className="bg-gray-900 text-gray-300 border-t border-gray-800 mt-auto">
  <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
    <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
      {/* About Section */}
      <div>
        <h3 className="text-white font-semibold mb-3">Goa Police</h3>
        <p className="text-sm text-gray-400 leading-relaxed">
          Digital Forensics Platform for voice analysis, biometric verification, and synthetic audio detection.
        </p>
      </div>
      
      {/* Quick Links */}
      <div>
        <h3 className="text-white font-semibold mb-3">Quick Links</h3>
        <ul className="space-y-2 text-sm">
          <li><a href="/dashboard" className="hover:text-violet-400 transition-colors">Dashboard</a></li>
          <li><a href="/voice-biometric" className="hover:text-violet-400 transition-colors">Voice Analysis</a></li>
          <li><a href="/synthetic-detection" className="hover:text-violet-400 transition-colors">Synthetic Detection</a></li>
        </ul>
      </div>
      
      {/* Legal */}
      <div>
        <h3 className="text-white font-semibold mb-3">Legal & Security</h3>
        <ul className="space-y-2 text-sm">
          <li className="flex items-center">
            <div className="w-2 h-2 bg-green-500 rounded-full mr-2"></div>
            Forensic-Grade Analysis
          </li>
          <li className="flex items-center">
            <div className="w-2 h-2 bg-green-500 rounded-full mr-2"></div>
            Court Admissible Reports
          </li>
          <li className="flex items-center">
            <div className="w-2 h-2 bg-green-500 rounded-full mr-2"></div>
            End-to-End Encryption
          </li>
        </ul>
      </div>
    </div>
    
    {/* Bottom Bar */}
    <div className="border-t border-gray-800 mt-8 pt-6 text-sm text-center text-gray-500">
      <p>&copy; 2025 Goa Police Department. All rights reserved. For official law enforcement use only.</p>
    </div>
  </div>
</footer>

    </div>
  );
};

export default Dashboard;

