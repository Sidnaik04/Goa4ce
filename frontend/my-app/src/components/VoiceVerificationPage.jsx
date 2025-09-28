import React, { useState, useRef } from 'react';
import { ArrowLeft, Upload, CheckCircle, XCircle, Download, Loader2, AlertTriangle } from 'lucide-react';

const VoiceVerificationPage = () => {
  const [inmateCode, setInmateCode] = useState('');
  const [selectedFile, setSelectedFile] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const fileInputRef = useRef(null);

  // API Base URL - should match your FastAPI server
  const API_BASE_URL = 'http://localhost:8000';

  const allowedFormats = ['.wav', '.mp3', '.flac', '.m4a', '.ogg'];
  
  const validateFile = (file) => {
    if (!file) return false;
    
    const fileExtension = '.' + file.name.split('.').pop().toLowerCase();
    return allowedFormats.includes(fileExtension);
  };

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    setError('');
    
    if (file) {
      if (validateFile(file)) {
        setSelectedFile(file);
      } else {
        setError(`Invalid file format. Please select one of: ${allowedFormats.join(', ')}`);
        setSelectedFile(null);
        event.target.value = '';
      }
    }
  };

  const handleSubmit = async () => {
    if (!inmateCode.trim()) {
      setError('Please enter an inmate code');
      return;
    }
    
    if (!selectedFile) {
      setError('Please select an audio file');
      return;
    }
    
    setIsLoading(true);
    setError('');
    setResult(null);
    
    try {
      const formData = new FormData();
      formData.append('audio', selectedFile);
      formData.append('inmate_code', inmateCode.trim());
      
      const token = localStorage.getItem('access_token');
      
      // Check if token exists
      if (!token) {
        throw new Error('No authentication token found. Please login again.');
      }
      
      console.log('Making request to:', `${API_BASE_URL}/upload`);
      console.log('Token exists:', !!token);
      
      const response = await fetch(`${API_BASE_URL}/upload`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
        },
        body: formData,
      });
      
      // Check if response has content before parsing JSON
      const contentType = response.headers.get('content-type');
      const responseText = await response.text();
      
      if (!response.ok) {
        let errorMessage = 'Verification failed';
        
        // Try to parse error response if it's JSON
        if (contentType && contentType.includes('application/json') && responseText) {
          try {
            const errorData = JSON.parse(responseText);
            errorMessage = errorData.detail || errorMessage;
          } catch (parseError) {
            errorMessage = responseText || errorMessage;
          }
        }
        
        throw new Error(errorMessage);
      }
      
      // Parse successful response
      if (!responseText) {
        throw new Error('Empty response from server');
      }
      
      let data;
      try {
        data = JSON.parse(responseText);
      } catch (parseError) {
        throw new Error('Invalid JSON response from server');
      }
      
      setResult(data);
      
    } catch (err) {
      console.error('Upload error:', err);
      setError(err.message || 'An error occurred during verification');
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setInmateCode('');
    setSelectedFile(null);
    setResult(null);
    setError('');
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleDownloadReport = async () => {
    if (!result?.pdf_report_url) return;
    
    try {
      const token = localStorage.getItem('access_token');
      const response = await fetch(`${API_BASE_URL}${result.pdf_report_url}`, {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });
      
      if (!response.ok) {
        setError('Failed to download report');
        return;
      }
      
      // Create blob from response
      const blob = await response.blob();
      
      // Create download link
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `forensic_report_${result.report_code}.pdf`;
      document.body.appendChild(link);
      link.click();
      
      // Cleanup
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
      
    } catch (err) {
      console.error('Download error:', err);
      setError('Failed to download report');
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-white to-violet-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b border-violet-100">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center space-x-4">
              <button
                onClick={() => window.history.back()}
                className="p-2 text-violet-600 hover:text-violet-700 hover:bg-violet-50 rounded-lg transition-colors"
              >
                <ArrowLeft className="w-5 h-5" />
              </button>
              <div>
                <h1 className="text-xl font-semibold text-gray-900">
                  Goa Police - Voice Verification
                </h1>
                <p className="text-sm text-gray-500">
                  Verify voice samples against registered inmates
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="bg-white rounded-2xl shadow-xl border border-violet-100 overflow-hidden">
          {/* Form Section */}
          <div className="p-8">
            <div className="mb-8">
              <h2 className="text-2xl font-bold text-gray-900 mb-2">
                Voice Sample Verification
              </h2>
              <p className="text-gray-600">
                Upload a voice sample and enter the inmate code to verify identity
              </p>
            </div>

            <div className="space-y-6">
              {/* Inmate Code Input */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Inmate Code
                </label>
                <input
                  type="text"
                  value={inmateCode}
                  onChange={(e) => setInmateCode(e.target.value)}
                  placeholder="e.g., INM001, INM002..."
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-violet-500 focus:border-violet-500 transition-colors"
                  disabled={isLoading}
                />
              </div>

              {/* File Upload */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Voice Sample
                </label>
                <div className="relative">
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".wav,.mp3,.flac,.m4a,.ogg"
                    onChange={handleFileSelect}
                    className="hidden"
                    disabled={isLoading}
                  />
                  <button
                    type="button"
                    onClick={() => fileInputRef.current?.click()}
                    className="w-full flex items-center justify-center px-6 py-4 border-2 border-dashed border-gray-300 rounded-lg hover:border-violet-400 hover:bg-violet-50 transition-colors"
                    disabled={isLoading}
                  >
                    <Upload className="w-8 h-8 text-gray-400 mb-2" />
                    <div className="ml-4 text-left">
                      <div className="text-sm font-medium text-gray-900">
                        {selectedFile ? selectedFile.name : 'Select audio file'}
                      </div>
                      <div className="text-xs text-gray-500">
                        Supported: {allowedFormats.join(', ')}
                      </div>
                    </div>
                  </button>
                </div>
              </div>

              {/* Error Display */}
              {error && (
                <div className="flex items-center space-x-3 p-4 bg-red-50 border border-red-200 rounded-lg">
                  <AlertTriangle className="w-5 h-5 text-red-500 flex-shrink-0" />
                  <span className="text-sm text-red-700">{error}</span>
                </div>
              )}

              {/* Submit Button */}
              <div className="flex space-x-4">
                <button
                  type="button"
                  onClick={handleSubmit}
                  disabled={isLoading || !inmateCode.trim() || !selectedFile}
                  className="flex-1 flex items-center justify-center px-6 py-3 bg-violet-600 text-white font-medium rounded-lg hover:bg-violet-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
                >
                  {isLoading ? (
                    <>
                      <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                      Verifying...
                    </>
                  ) : (
                    'Verify Voice Sample'
                  )}
                </button>
                
                <button
                  type="button"
                  onClick={handleReset}
                  disabled={isLoading}
                  className="px-6 py-3 bg-gray-100 text-gray-700 font-medium rounded-lg hover:bg-gray-200 disabled:opacity-50 transition-colors"
                >
                  Reset
                </button>
              </div>
            </div>
          </div>

          {/* Results Section */}
          {result && (
            <div className="border-t border-gray-200 bg-gray-50 p-8">
              <h3 className="text-lg font-semibold text-gray-900 mb-6">
                Verification Results
              </h3>
              
              <div className="grid md:grid-cols-2 gap-6">
                {/* Voice Match Result */}
                <div className="bg-white rounded-lg p-6 border border-gray-200">
                  <div className="flex items-center space-x-3 mb-4">
                    {result.speaker_match ? (
                      <CheckCircle className="w-8 h-8 text-green-500" />
                    ) : (
                      <XCircle className="w-8 h-8 text-red-500" />
                    )}
                    <div>
                      <h4 className="font-semibold text-gray-900">
                        {result.speaker_match ? 'Voice Match Found' : 'No Match Found'}
                      </h4>
                      <p className="text-sm text-gray-600">
                        Confidence: {(result.speaker_match_confidence * 100).toFixed(1)}%
                      </p>
                    </div>
                  </div>
                  
                  {result.speaker_match && result.matched_inmate_code && (
                    <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                      <p className="text-sm font-medium text-green-800">
                        Matched Inmate: {result.matched_inmate_code}
                      </p>
                    </div>
                  )}
                </div>

              </div>

              {/* Report Information */}
              <div className="mt-6 bg-white rounded-lg p-6 border border-gray-200">
                <div className="flex items-center justify-between mb-4">
                  <div>
                    <h4 className="font-semibold text-gray-900">Forensic Report</h4>
                    <p className="text-sm text-gray-600">
                      Report Code: {result.report_code}
                    </p>
                  </div>
                  <button
                    onClick={handleDownloadReport}
                    className="flex items-center space-x-2 px-4 py-2 bg-violet-600 text-white font-medium rounded-lg hover:bg-violet-700 transition-colors"
                  >
                    <Download className="w-4 h-4" />
                    <span>Download PDF</span>
                  </button>
                </div>
                
                <div className="text-xs text-gray-500">
                  This report contains detailed analysis including audio metadata, 
                  spectrograms, and verification signatures for legal admissibility.
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default VoiceVerificationPage;