import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Shield, ArrowLeft, CheckCircle, User, FileAudio, Calendar, Database } from 'lucide-react';

const VoiceBiometricResults = () => {
  const navigate = useNavigate();
  const [resultData, setResultData] = useState(null);

  useEffect(() => {
    // Get the result data from localStorage
    const storedResult = localStorage.getItem('inmateCreationResult');
    if (storedResult) {
      setResultData(JSON.parse(storedResult));
    } else {
      // If no result data, redirect back to the form
      navigate('/voice-biometric');
    }
  }, [navigate]);

  const goBack = () => {
    navigate('/voice-biometric');
  };

  const goToDashboard = () => {
    // Clear the stored result
    localStorage.removeItem('inmateCreationResult');
    navigate('/dashboard');
  };

  if (!resultData) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600"></div>
      </div>
    );
  }

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleString();
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center space-x-3">
              <button
                onClick={goBack}
                className="flex items-center space-x-2 text-gray-600 hover:text-gray-800 transition-colors duration-200"
              >
                <ArrowLeft className="h-5 w-5" />
                <span>Back to Form</span>
              </button>
            </div>
            
            <div className="flex items-center space-x-3">
              <div className="bg-indigo-100 p-2 rounded-full">
                <Shield className="h-8 w-8 text-indigo-600" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-gray-900">GOA POLICE</h1>
                <p className="text-sm text-gray-600">Registration Complete</p>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {/* Success Header */}
        <div className="text-center mb-12">
          <div className="flex justify-center mb-6">
            <div className="bg-green-100 p-4 rounded-full">
              <CheckCircle className="h-16 w-16 text-green-600" />
            </div>
          </div>
          <h2 className="text-4xl font-bold text-gray-900 mb-4">
            Profile Created Successfully!
          </h2>
          <p className="text-lg text-gray-600">
            Voice biometric profile has been registered in the system
          </p>
        </div>

        {/* Results Card */}
        <div className="bg-white rounded-2xl shadow-lg border border-gray-200 overflow-hidden mb-8">
          <div className="bg-gradient-to-r from-indigo-50 to-purple-50 px-8 py-6 border-b border-gray-200">
            <h3 className="text-2xl font-bold text-gray-900">Registration Details</h3>
            <p className="text-gray-600 mt-1">Inmate profile and voiceprint information</p>
          </div>

          <div className="p-8">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              {/* Profile Information */}
              <div className="space-y-6">
                <h4 className="text-lg font-semibold text-gray-800 flex items-center">
                  <User className="h-5 w-5 mr-2 text-indigo-600" />
                  Profile Information
                </h4>
                
                <div className="space-y-4">
                  <div className="bg-gray-50 p-4 rounded-lg">
                    <label className="text-sm font-medium text-gray-500">Profile ID</label>
                    <p className="text-lg font-semibold text-gray-900">{resultData.id}</p>
                  </div>
                  
                  <div className="bg-gray-50 p-4 rounded-lg">
                    <label className="text-sm font-medium text-gray-500">Full Name</label>
                    <p className="text-lg font-semibold text-gray-900">{resultData.name}</p>
                  </div>
                  
                  <div className="bg-gray-50 p-4 rounded-lg">
                    <label className="text-sm font-medium text-gray-500">Inmate Code</label>
                    <p className="text-lg font-semibold text-gray-900">{resultData.inmate_code}</p>
                  </div>
                </div>
              </div>

              {/* Voiceprint Information */}
              <div className="space-y-6">
                <h4 className="text-lg font-semibold text-gray-800 flex items-center">
                  <FileAudio className="h-5 w-5 mr-2 text-purple-600" />
                  Voiceprint Information
                </h4>
                
                <div className="space-y-4">
                  <div className="bg-gray-50 p-4 rounded-lg">
                    <label className="text-sm font-medium text-gray-500">Voiceprint ID</label>
                    <p className="text-lg font-semibold text-gray-900">{resultData.voiceprint_id}</p>
                  </div>
                  
                  <div className="bg-gray-50 p-4 rounded-lg">
                    <label className="text-sm font-medium text-gray-500">Created At</label>
                    <p className="text-lg font-semibold text-gray-900">
                      {formatDate(resultData.created_at)}
                    </p>
                  </div>
                  
                  <div className="bg-green-50 p-4 rounded-lg border border-green-200">
                    <label className="text-sm font-medium text-green-700">Status</label>
                    <p className="text-lg font-semibold text-green-800">Active & Ready for Analysis</p>
                  </div>
                </div>
              </div>
            </div>

            {/* Additional Information */}
            <div className="mt-8 pt-8 border-t border-gray-200">
              <h4 className="text-lg font-semibold text-gray-800 flex items-center mb-4">
                <Database className="h-5 w-5 mr-2 text-indigo-600" />
                System Information
              </h4>
              
              <div className="bg-indigo-50 p-6 rounded-lg">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                  <div className="flex items-center">
                    <div className="w-2 h-2 bg-indigo-500 rounded-full mr-3"></div>
                    <span className="text-indigo-700">Voice features extracted</span>
                  </div>
                  <div className="flex items-center">
                    <div className="w-2 h-2 bg-indigo-500 rounded-full mr-3"></div>
                    <span className="text-indigo-700">Biometric signature created</span>
                  </div>
                  <div className="flex items-center">
                    <div className="w-2 h-2 bg-indigo-500 rounded-full mr-3"></div>
                    <span className="text-indigo-700">Profile stored securely</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Next Steps */}
        <div className="bg-white rounded-2xl shadow-lg border border-gray-200 p-8 mb-8">
          <h3 className="text-xl font-bold text-gray-900 mb-4 flex items-center">
            <Calendar className="h-6 w-6 mr-2 text-purple-600" />
            Next Steps
          </h3>
          
          <div className="space-y-3 text-gray-700">
            <div className="flex items-start">
              <div className="w-6 h-6 bg-purple-100 rounded-full flex items-center justify-center mr-3 mt-0.5">
                <span className="text-xs font-semibold text-purple-600">1</span>
              </div>
              <p>The voice profile is now available for speaker verification and identification tasks</p>
            </div>
            
            <div className="flex items-start">
              <div className="w-6 h-6 bg-purple-100 rounded-full flex items-center justify-center mr-3 mt-0.5">
                <span className="text-xs font-semibold text-purple-600">2</span>
              </div>
              <p>Upload unknown audio samples to compare against this profile</p>
            </div>
            
            <div className="flex items-start">
              <div className="w-6 h-6 bg-purple-100 rounded-full flex items-center justify-center mr-3 mt-0.5">
                <span className="text-xs font-semibold text-purple-600">3</span>
              </div>
              <p>Generate forensic reports with confidence scores and digital signatures</p>
            </div>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex flex-col sm:flex-row gap-4 justify-center">
          <button
            onClick={goBack}
            className="px-8 py-3 border-2 border-indigo-600 text-indigo-600 font-medium rounded-lg hover:bg-indigo-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 transition-all duration-200"
          >
            Register Another Profile
          </button>
          
          <button
            onClick={goToDashboard}
            className="px-8 py-3 bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700 text-white font-medium rounded-lg focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 transition-all duration-200 transform hover:scale-105"
          >
            Return to Dashboard
          </button>
        </div>
      </main>
    </div>
  );
};

export default VoiceBiometricResults;