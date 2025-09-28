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
                <span>Back to Dashboard</span>
              </button>
            </div>
            
            <div className="flex items-center space-x-3">
              <div className="bg-indigo-100 p-2 rounded-full">
                <Shield className="h-8 w-8 text-indigo-600" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-gray-900">GOA POLICE</h1>
                <p className="text-sm text-gray-600">Voice Biometric Registration</p>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="text-center mb-12">
          <div className="flex justify-center mb-6">
            <div className="bg-gradient-to-br from-indigo-500 to-purple-600 p-4 rounded-2xl">
              <Mic className="h-12 w-12 text-white" />
            </div>
          </div>
          <h2 className="text-4xl font-bold text-gray-900 mb-4">
            Register New Inmate Profile
          </h2>
          <p className="text-lg text-gray-600">
            Create a voice biometric profile for speaker identification
          </p>
        </div>

        {/* Form */}
        <div className="bg-white rounded-2xl shadow-lg border border-gray-200 overflow-hidden">
          <div className="p-8">
            {/* Success Message */}
            {success && (
              <div className="bg-green-50 border border-green-200 rounded-lg p-4 mb-6">
                <div className="flex items-center">
                  <CheckCircle className="h-5 w-5 text-green-500 mr-3" />
                  <div>
                    <h3 className="text-sm font-medium text-green-800">Success!</h3>
                    <div className="mt-1 text-sm text-green-700">{success}</div>
                  </div>
                </div>
              </div>
            )}

            {/* Error Message */}
            {error && (
              <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
                <div className="flex items-center">
                  <AlertCircle className="h-5 w-5 text-red-500 mr-3" />
                  <div>
                    <h3 className="text-sm font-medium text-red-800">Error</h3>
                    <div className="mt-1 text-sm text-red-700">{error}</div>
                  </div>
                </div>
              </div>
            )}

            <form onSubmit={handleSubmit} className="space-y-6">
              {/* Name Field */}
              <div>
                <label htmlFor="name" className="block text-sm font-medium text-gray-700 mb-2">
                  Full Name
                </label>
                <div className="relative">
                  <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                    <User className="h-5 w-5 text-gray-400" />
                  </div>
                  <input
                    type="text"
                    id="name"
                    name="name"
                    value={formData.name}
                    onChange={handleInputChange}
                    className={`block w-full pl-10 pr-3 py-3 border rounded-lg placeholder-gray-500 text-gray-900 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 bg-white transition-all duration-200 ${
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

              {/* Inmate Code Field */}
              <div>
                <label htmlFor="inmate_code" className="block text-sm font-medium text-gray-700 mb-2">
                  Inmate Code
                </label>
                <div className="relative">
                  <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                    <Shield className="h-5 w-5 text-gray-400" />
                  </div>
                  <input
                    type="text"
                    id="inmate_code"
                    name="inmate_code"
                    value={formData.inmate_code}
                    onChange={handleInputChange}
                    className={`block w-full pl-10 pr-3 py-3 border rounded-lg placeholder-gray-500 text-gray-900 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 bg-white transition-all duration-200 ${
                      validationErrors.inmate_code ? 'border-red-300' : 'border-gray-300'
                    }`}
                    placeholder="INM001"
                    disabled={isLoading}
                  />
                </div>
                <p className="mt-1 text-xs text-gray-500">Format: INM + 3 digits (e.g., INM001, INM123)</p>
                {validationErrors.inmate_code && (
                  <p className="mt-1 text-sm text-red-600">{validationErrors.inmate_code}</p>
                )}
              </div>

              {/* Reference Audio Field */}
              <div>
                <label htmlFor="reference_audio" className="block text-sm font-medium text-gray-700 mb-2">
                  Reference Audio File
                </label>
                <div className="relative">
                  <div className="flex items-center justify-center w-full">
                    <label
                      htmlFor="reference_audio"
                      className={`flex flex-col items-center justify-center w-full h-32 border-2 border-dashed rounded-lg cursor-pointer bg-gray-50 hover:bg-gray-100 transition-colors duration-200 ${
                        validationErrors.reference_audio ? 'border-red-300' : 'border-gray-300'
                      }`}
                    >
                      <div className="flex flex-col items-center justify-center pt-5 pb-6">
                        {formData.reference_audio ? (
                          <>
                            <FileAudio className="w-8 h-8 mb-3 text-indigo-500" />
                            <p className="text-sm text-gray-700 font-medium">{formData.reference_audio.name}</p>
                            <p className="text-xs text-gray-500">Click to change file</p>
                          </>
                        ) : (
                          <>
                            <Upload className="w-8 h-8 mb-3 text-gray-400" />
                            <p className="mb-2 text-sm text-gray-500">
                              <span className="font-semibold">Click to upload</span> or drag and drop
                            </p>
                            <p className="text-xs text-gray-500">WAV, MP3, FLAC, M4A, OGG</p>
                          </>
                        )}
                      </div>
                      <input
                        id="reference_audio"
                        type="file"
                        accept=".wav,.mp3,.flac,.m4a,.ogg,audio/*"
                        onChange={handleFileChange}
                        className="hidden"
                        disabled={isLoading}
                      />
                    </label>
                  </div>
                </div>
                {validationErrors.reference_audio && (
                  <p className="mt-1 text-sm text-red-600">{validationErrors.reference_audio}</p>
                )}
              </div>

              {/* Submit Button */}
              <div className="flex justify-end space-x-4 pt-6">
                <button
                  type="button"
                  onClick={goBack}
                  className="px-6 py-3 border border-gray-300 rounded-lg text-gray-700 font-medium hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 transition-all duration-200"
                  disabled={isLoading}
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={isLoading}
                  className="px-8 py-3 bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700 text-white font-medium rounded-lg focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 transform hover:scale-105"
                >
                  {isLoading ? (
                    <div className="flex items-center">
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                      Creating Profile...
                    </div>
                  ) : (
                    'Create Profile'
                  )}
                </button>
              </div>
            </form>
          </div>
        </div>

        {/* Information Panel */}
        <div className="mt-8 bg-indigo-50 rounded-xl p-6 border border-indigo-200">
          <h3 className="text-lg font-semibold text-indigo-800 mb-3">
            Voice Biometric Registration Process
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-indigo-700">
            <div className="flex items-start">
              <div className="w-2 h-2 bg-indigo-500 rounded-full mr-3 mt-2 flex-shrink-0"></div>
              <p>Upload a clear audio sample of the subject's voice (minimum 10 seconds recommended)</p>
            </div>
            <div className="flex items-start">
              <div className="w-2 h-2 bg-indigo-500 rounded-full mr-3 mt-2 flex-shrink-0"></div>
              <p>System will extract voice features and create a unique biometric signature</p>
            </div>
            <div className="flex items-start">
              <div className="w-2 h-2 bg-indigo-500 rounded-full mr-3 mt-2 flex-shrink-0"></div>
              <p>Profile can be used for future voice verification and identification</p>
            </div>
            <div className="flex items-start">
              <div className="w-2 h-2 bg-indigo-500 rounded-full mr-3 mt-2 flex-shrink-0"></div>
              <p>All data is encrypted and stored securely for forensic use</p>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
};

export default VoiceBiometricPage;