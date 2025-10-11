import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Shield, ArrowLeft, Mic, Upload, User, FileAudio, AlertCircle, CheckCircle } from 'lucide-react';

const VoiceBiometricPage = () => {
  const navigate = useNavigate();
  const [formData, setFormData] = useState({
    name: '',
    inmate_code: '',
    reference_audio: null
  });
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [validationErrors, setValidationErrors] = useState({});

  const API_BASE_URL = 'http://localhost:8000';

  const goBack = () => {
    navigate('/dashboard');
  };

  const validateInmateCode = (code) => {
    const regex = /^INM\d{3}$/;
    return regex.test(code);
  };

  const validateForm = () => {
    const errors = {};

    if (!formData.name.trim()) {
      errors.name = 'Name is required';
    }

    if (!formData.inmate_code.trim()) {
      errors.inmate_code = 'Inmate code is required';
    } else if (!validateInmateCode(formData.inmate_code)) {
      errors.inmate_code = 'Inmate code must follow format INM + 3 digits (e.g., INM001)';
    }

    if (!formData.reference_audio) {
      errors.reference_audio = 'Reference audio file is required';
    } else {
      const allowedTypes = ['.wav', '.mp3', '.flac', '.m4a', '.ogg'];
      const fileExtension = '.' + formData.reference_audio.name.split('.').pop().toLowerCase();
      if (!allowedTypes.includes(fileExtension)) {
        errors.reference_audio = 'Please select a valid audio file (.wav, .mp3, .flac, .m4a, .ogg)';
      }
    }

    setValidationErrors(errors);
    return Object.keys(errors).length === 0;
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
    
    // Clear validation error for this field
    if (validationErrors[name]) {
      setValidationErrors(prev => ({
        ...prev,
        [name]: ''
      }));
    }
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    setFormData(prev => ({
      ...prev,
      reference_audio: file
    }));
    
    // Clear validation error for file
    if (validationErrors.reference_audio) {
      setValidationErrors(prev => ({
        ...prev,
        reference_audio: ''
      }));
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setSuccess('');

    if (!validateForm()) {
      return;
    }

    setIsLoading(true);

    try {
      const token = localStorage.getItem('access_token');
      const tokenType = localStorage.getItem('token_type') || 'bearer';

      if (!token) {
        navigate('/login');
        return;
      }

      // Create FormData
      const submitData = new FormData();
      submitData.append('name', formData.name);
      submitData.append('inmate_code', formData.inmate_code);
      submitData.append('reference_audio', formData.reference_audio);

      const response = await fetch(`${API_BASE_URL}/inmates`, {
        method: 'POST',
        headers: {
          'Authorization': `${tokenType} ${token}`,
        },
        body: submitData,
      });

      const data = await response.json();

      if (response.ok) {
        setSuccess(`Inmate profile created successfully! ID: ${data.id}, Voiceprint ID: ${data.voiceprint_id}`);
        
        // Store the result data for the results page
        localStorage.setItem('inmateCreationResult', JSON.stringify(data));
        
        // Redirect to results page after 2 seconds
        setTimeout(() => {
          navigate('/voice-biometric/results');
        }, 2000);

      } else {
        setError(data.detail || 'Failed to create inmate profile. Please try again.');
      }

    } catch (error) {
      console.error('Submission error:', error);
      setError('Network error. Please check your connection and try again.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-white">
      {/* Navbar */}
      {/* Navbar */}
<header className="bg-white shadow-sm border-b border-violet-100">
  <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
    <div className="flex items-center justify-between h-16">
      <div className="flex items-center space-x-4">
        <button
          onClick={goBack}
          className="p-2 text-violet-600 hover:text-violet-700 hover:bg-violet-50 rounded-lg transition-colors"
        >
          <ArrowLeft className="w-5 h-5" />
        </button>
        <div>
          <h1 className="text-xl font-semibold text-gray-900">
            Goa Police - Inmate Biometric Registration
          </h1>
          <p className="text-sm text-gray-500">
            Register new inmates with voice biometrics
          </p>
        </div>
      </div>
    </div>
  </div>
</header>


      {/* Main Content */}
      <main className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 pt-12 pb-16">
        <div className="text-center mb-12">
  <div className="flex justify-center mb-6">
    <img 
      src="/inmate.png" 
      alt="Inmate Registration" 
      className="h-32 w-auto object-contain" 
    />
  </div>
  <h2 className="text-3xl sm:text-4xl font-extrabold text-gray-900 mb-4">
    Register New Inmate Profile
  </h2>
  <p className="text-gray-600 text-lg">Create a voice biometric profile for identification</p>
</div>

        {/* Form Card */}
        <div className="bg-white rounded-2xl shadow-lg border-2 border-purple-300 
                hover:border-purple-500 transition-all duration-200">
          <div className="p-8">
            {/* Success */}
            {success && (
              <div className="bg-green-50 border border-green-200 rounded-lg p-4 mb-6 flex items-start">
                <CheckCircle className="h-5 w-5 text-green-600 mr-3 mt-1" />
                <div>
                  <h3 className="text-sm font-semibold text-green-800">Success</h3>
                  <p className="text-sm text-green-700">{success}</p>
                </div>
              </div>
            )}
            {/* Error */}
            {error && (
              <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6 flex items-start">
                <AlertCircle className="h-5 w-5 text-red-600 mr-3 mt-1" />
                <div>
                  <h3 className="text-sm font-semibold text-red-800">Error</h3>
                  <p className="text-sm text-red-700">{error}</p>
                </div>
              </div>
            )}

            <form onSubmit={handleSubmit} className="space-y-6">
              {/* Name */}
              <div>
                <label htmlFor="name" className="block text-sm font-medium text-gray-700 mb-2">
                  Full Name
                </label>
                <div className="relative">
                  <User className="h-5 w-5 text-gray-400 absolute left-3 top-3" />
                  <input
                    type="text"
                    id="name"
                    name="name"
                    value={formData.name}
                    onChange={handleInputChange}
                    className={`w-full pl-10 pr-3 py-3 border rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 transition ${
                      validationErrors.name ? 'border-red-300' : 'border-gray-300'
                    }`}
                    placeholder="Enter full name"
                    disabled={isLoading}
                  />
                </div>
                {validationErrors.name && (
                  <p className="mt-1 text-sm text-red-600">{validationErrors.name}</p>
                )}
              </div>

              {/* Inmate Code */}
              <div>
                <label htmlFor="inmate_code" className="block text-sm font-medium text-gray-700 mb-2">
                  Inmate Code
                </label>
                <div className="relative">
                  <Shield className="h-5 w-5 text-gray-400 absolute left-3 top-3" />
                  <input
                    type="text"
                    id="inmate_code"
                    name="inmate_code"
                    value={formData.inmate_code}
                    onChange={handleInputChange}
                    className={`w-full pl-10 pr-3 py-3 border rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 transition ${
                      validationErrors.inmate_code ? 'border-red-300' : 'border-gray-300'
                    }`}
                    placeholder="INM001"
                    disabled={isLoading}
                  />
                </div>
                <p className="mt-1 text-xs text-gray-500">Format: INM + 3 digits (e.g., INM001)</p>
                {validationErrors.inmate_code && (
                  <p className="mt-1 text-sm text-red-600">{validationErrors.inmate_code}</p>
                )}
              </div>

              {/* File Upload */}
              <div>
                <label htmlFor="reference_audio" className="block text-sm font-medium text-gray-700 mb-2">
                  Reference Audio File
                </label>
                <label
                  htmlFor="reference_audio"
                  className={`flex flex-col items-center justify-center w-full h-32 border-2 border-dashed rounded-lg cursor-pointer hover:bg-gray-50 transition ${
                    validationErrors.reference_audio ? 'border-red-300' : 'border-gray-300'
                  }`}
                >
                  {formData.reference_audio ? (
                    <>
                      <FileAudio className="w-8 h-8 mb-2 text-indigo-600" />
                      <p className="text-sm text-gray-700 font-medium">{formData.reference_audio.name}</p>
                      <p className="text-xs text-gray-500">Click to change file</p>
                    </>
                  ) : (
                    <>
                      <Upload className="w-8 h-8 mb-2 text-gray-400" />
                      <p className="text-sm text-gray-500">Click to upload or drag & drop</p>
                      <p className="text-xs text-gray-500">WAV, MP3, FLAC, M4A, OGG</p>
                    </>
                  )}
                  <input
                    id="reference_audio"
                    type="file"
                    accept=".wav,.mp3,.flac,.m4a,.ogg,audio/*"
                    onChange={handleFileChange}
                    className="hidden"
                    disabled={isLoading}
                  />
                </label>
                {validationErrors.reference_audio && (
                  <p className="mt-1 text-sm text-red-600">{validationErrors.reference_audio}</p>
                )}
              </div>

              {/* Actions */}
              <div className="flex justify-end space-x-4 pt-6">
                <button
                  type="button"
                  onClick={goBack}
                  className="px-6 py-3 border border-gray-300 rounded-lg text-gray-700 font-medium hover:bg-gray-50 transition"
                  disabled={isLoading}
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={isLoading}
                  className="px-8 py-3 bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700 text-white font-medium rounded-lg focus:ring-2 focus:ring-indigo-500 disabled:opacity-50 transition transform hover:scale-105"
                >
                  {isLoading ? (
                    <div className="flex items-center">
                      <div className="animate-spin h-4 w-4 border-b-2 border-white rounded-full mr-2"></div>
                      Creating...
                    </div>
                  ) : (
                    'Create Profile'
                  )}
                </button>
              </div>
            </form>
          </div>
        </div>

        {/* Info Box */}
        <div className="mt-12 bg-indigo-50 rounded-xl p-6 border border-indigo-200">
          <h3 className="text-lg font-semibold text-indigo-800 mb-3">Registration Process</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-indigo-700">
            <div className="flex items-start">
              <div className="w-2 h-2 bg-indigo-500 rounded-full mr-2 mt-2"></div>
              <p>Upload a clear 10+ second audio sample of the inmate's voice.</p>
            </div>
            <div className="flex items-start">
              <div className="w-2 h-2 bg-indigo-500 rounded-full mr-2 mt-2"></div>
              <p>The system extracts features and creates a unique biometric signature.</p>
            </div>
            <div className="flex items-start">
              <div className="w-2 h-2 bg-indigo-500 rounded-full mr-2 mt-2"></div>
              <p>Voiceprint can be used for future verification and identification.</p>
            </div>
            <div className="flex items-start">
              <div className="w-2 h-2 bg-indigo-500 rounded-full mr-2 mt-2"></div>
              <p>All data is encrypted and securely stored for forensic use.</p>
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

export default VoiceBiometricPage;

