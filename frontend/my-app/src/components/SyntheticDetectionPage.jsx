import React, { useState } from 'react';
import { Shield, ArrowLeft, Waves, Upload, FileAudio, AlertCircle, CheckCircle, X, Download, Clock, Cpu, BarChart3 } from 'lucide-react';

const SyntheticDetectionPage = () => {
  const [audioFile, setAudioFile] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  const API_BASE_URL = 'http://localhost:3000'; // Your deepfake detection server
  const AUTH_API_URL = 'http://localhost:8000'; // Your main authentication server

  const goBack = () => {
    window.history.back();
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      const allowedTypes = ['.wav', '.mp3', '.flac', '.m4a', '.ogg'];
      const fileExtension = '.' + file.name.split('.').pop().toLowerCase();
      
      if (!allowedTypes.includes(fileExtension)) {
        setError('Please select a valid audio file (.wav, .mp3, .flac, .m4a, .ogg)');
        setAudioFile(null);
        return;
      }
      
      setAudioFile(file);
      setError('');
      setResult(null);
    }
  };

  const analyzeAudio = async () => {
    if (!audioFile) {
      setError('Please select an audio file to analyze');
      return;
    }

    setIsAnalyzing(true);
    setError('');
    setResult(null);

    try {
      const formData = new FormData();
      formData.append('file', audioFile);

      // Try multiple possible endpoint variations
      const endpoints = [
        `${API_BASE_URL}/api/v1/predict/`,
        `${API_BASE_URL}/api/v1/predict`
      ];


      console.log('Trying endpoints:', endpoints);
      console.log('File:', audioFile.name);

      let response;
      let lastError;

      for (const endpoint of endpoints) {
        try {
          console.log(`Trying: ${endpoint}`);
          response = await fetch(endpoint, {
            method: 'POST',
            body: formData,
          });

          if (response.ok) {
            console.log(`Success with endpoint: ${endpoint}`);
            break;
          } else {
            console.log(`Failed with ${endpoint}: ${response.status} ${response.statusText}`);
            lastError = `${response.status}: ${response.statusText}`;
          }
        } catch (err) {
          console.log(`Error with ${endpoint}:`, err.message);
          lastError = err.message;
        }
      }

      if (!response || !response.ok) {
        throw new Error(lastError || 'All endpoints failed');
      }

      const data = await response.json();
      console.log('API Response:', data);
      
      setResult(data.report_data);

    } catch (error) {
      console.error('Analysis error:', error);
      setError(`Analysis failed: ${error.message}. Please check if your server is running on port 3000 and the endpoint is configured correctly.`);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const downloadReport = async () => {
    if (!result) return;

    try {
      // Create a comprehensive report
      const reportData = {
        report_id: result.report_id,
        analysis_timestamp: result.date_of_analysis,
        file_metadata: result.audio_metadata,
        analysis_results: {
          final_verdict: result.final_verdict,
          confidence_scores: {
            genuine: result.average_genuine_score,
            deepfake: result.average_deepfake_score
          },
          chunks_analyzed: result.chunks_analyzed,
          detailed_chunks: result.chunk_results
        },
        technical_info: {
          models_used: result.tool_used,
          device: result.device_used,
          analyzed_by: result.analyzed_by
        }
      };

      const linkElement = document.createElement('a');

      // JSON Report Download
      const dataStr = JSON.stringify(reportData, null, 2);
      const dataUri = 'data:application/json;charset=utf-8,' + encodeURIComponent(dataStr);
      linkElement.setAttribute('href', dataUri);
      linkElement.setAttribute('download', `synthetic_detection_report_${result.report_id}.json`);
      linkElement.click();

      // PDF Report Download
      const pdfUrl = `${API_BASE_URL}/pdf_reports/${result.report_id}.pdf`;
      linkElement.setAttribute('href', pdfUrl);
      linkElement.setAttribute('download', `synthetic_detection_report_${result.report_id}.pdf`);
      linkElement.click();

    } catch (error) {
      console.error('Download error:', error);
      setError('Failed to download report. Please try again.');
    }
  };

  const resetAnalysis = () => {
    setAudioFile(null);
    setResult(null);
    setError('');
  };

  const getVerdictColor = (verdict) => {
    if (verdict === 'Human Voice') return 'green';
    if (verdict === 'AI / Synthetic Voice') return 'red';
    if (verdict === 'Uncertain') return 'yellow';
    return 'gray';
  };

  const getVerdictIcon = (verdict) => {
    if (verdict === 'Human Voice') return <CheckCircle className="h-16 w-16 text-green-600" />;
    if (verdict === 'AI / Synthetic Voice') return <X className="h-16 w-16 text-red-600" />;
    return <AlertCircle className="h-16 w-16 text-yellow-600" />;
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
                </button>
        <div>
          <h1 className="text-xl font-semibold text-gray-900">
            Goa Police - Synthetic Voice Detection
          </h1>
          <p className="text-sm text-gray-500">
            Audio and Deepfake Detection System
          </p>
        </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="text-center mb-12">
          <div className="flex justify-center mb-6">
            <img 
              src="/synthetic.png" 
              alt="Synthetic Voice Detection" 
              className="h-32 w-auto object-contain" 
            />
          </div>
          <h2 className="text-4xl font-bold text-gray-900 mb-4">
            Synthetic Voice Detection
          </h2>
          </div>

        {/* Analysis Form */}
         <div className="bg-white rounded-2xl shadow-lg border-2 border-purple-300 hover:border-purple-500 
                transition-all duration-200 overflow-hidden mb-8">
          <div className="p-8">
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

            {/* File Upload */}
            <div className="mb-8">
              <label htmlFor="audio-file" className="block text-sm font-medium text-gray-700 mb-2">
                Upload Audio File for Analysis
              </label>
              <div className="flex items-center justify-center w-full">
                <label
  htmlFor="audio-file"
  className="flex flex-col items-center justify-center w-full h-32 
             border-2 border-dashed border-gray-400 rounded-lg cursor-pointer 
             bg-gray-50 hover:bg-gray-100 
             transition-all duration-200
             hover:border-purple-500 hover:shadow-md"
>
                  <div className="flex flex-col items-center justify-center pt-5 pb-6">
                    {audioFile ? (
                      <>
                        <FileAudio className="w-8 h-8 mb-3 text-purple-500" />
                        <p className="text-sm text-gray-700 font-medium">{audioFile.name}</p>
                        <p className="text-xs text-gray-500">Click to change file</p>
                      </>
                    ) : (
                      <>
                        <Upload className="w-8 h-8 mb-3 text-gray-400" />
                        <p className="mb-2 text-sm text-gray-500">
                          <span className="font-semibold">Click to upload</span> or drag and drop
                        </p>
                        <p className="text-xs text-gray-500">WAV, MP3, FLAC, M4A, OGG (Max 50MB)</p>
                      </>
                    )}
                  </div>
                  <input
                    id="audio-file"
                    type="file"
                    accept=".wav,.mp3,.flac,.m4a,.ogg,audio/*"
                    onChange={handleFileChange}
                    className="hidden"
                    disabled={isAnalyzing}
                  />
                </label>
              </div>
            </div>

            {/* Analyze Button */}
            <div className="flex justify-center mb-8">
              <button
                onClick={analyzeAudio}
                disabled={!audioFile || isAnalyzing}
                className="px-8 py-3 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 text-white font-medium rounded-lg focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-purple-500 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 transform hover:scale-105"
              >
                {isAnalyzing ? (
                  <div className="flex items-center">
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                    Analyzing Audio...
                  </div>
                ) : (
                  'Analyze for Synthetic Voice'
                )}
              </button>
            </div>

            {/* Results Section */}
            {result && (
              <div className="border-t border-gray-200 pt-8">
                <div className="text-center">
                  <div className="flex justify-center mb-6">
                    <div className={`bg-${getVerdictColor(result.final_verdict)}-100 p-4 rounded-full`}>
                      {getVerdictIcon(result.final_verdict)}
                    </div>
                  </div>

                  <h3 className="text-2xl font-bold mb-2">
                    <span className={`text-${getVerdictColor(result.final_verdict)}-600`}>
                      {result.final_verdict}
                    </span>
                  </h3>

                  <p className="text-gray-600 mb-6">
                    Report ID: {result.report_id}
                  </p>


                  {/* Technical Details */}
                  <div className="bg-gray-50 rounded-lg p-6 mb-6 text-left max-w-2xl mx-auto">
                    <h4 className="text-lg font-semibold text-gray-900 mb-4 text-center">Analysis Details</h4>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                      <div className="flex items-center space-x-2">
                        <Clock className="h-4 w-4 text-gray-500" />
                        <span className="text-gray-600">Duration: {result.audio_metadata?.duration || 'N/A'}</span>
                      </div>
                      <div className="flex items-center space-x-2">
                        <BarChart3 className="h-4 w-4 text-gray-500" />
                        <span className="text-gray-600">Chunks: {result.chunks_analyzed}</span>
                      </div>
                      <div className="flex items-center space-x-2">
                        <Cpu className="h-4 w-4 text-gray-500" />
                        <span className="text-gray-600">Device: {result.device_used}</span>
                      </div>
                    </div>
                    
                    {/* File Metadata */}
                    {result.audio_metadata && (
                      <div className="mt-4 pt-4 border-t border-gray-200">
                        <h5 className="font-medium text-gray-900 mb-2">File Information</h5>
                        <div className="text-xs text-gray-600 space-y-1">
                          <p>Format: {result.audio_metadata.file_format?.toUpperCase()}</p>
                          <p>Sample Rate: {result.audio_metadata.sampling_rate}</p>
                          <p>MD5 Hash: {result.audio_metadata.md5_hash?.substring(0, 16)}...</p>
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Chunk Results */}
                  {result.chunk_results && result.chunk_results.length > 1 && (
                    <div className="bg-white border border-gray-200 rounded-lg p-4 mb-6 max-w-2xl mx-auto">
                      <h5 className="font-medium text-gray-900 mb-3">Chunk Analysis</h5>
                      <div className="space-y-2">
                        {result.chunk_results.slice(0, 5).map((chunk, index) => (
                          <div key={index} className="flex justify-between items-center text-sm">
                            <span className="text-gray-600">Chunk {chunk.chunk_index}</span>
                            <div className="flex space-x-4">
                              <span className="text-green-600">
                                G: {(chunk.genuine_score * 100).toFixed(1)}%
                              </span>
                              <span className="text-red-600">
                                D: {(chunk.deepfake_score * 100).toFixed(1)}%
                              </span>
                            </div>
                          </div>
                        ))}
                        {result.chunk_results.length > 5 && (
                          <p className="text-xs text-gray-500 text-center">
                            ... and {result.chunk_results.length - 5} more chunks
                          </p>
                        )}
                      </div>
                    </div>
                  )}

                  {/* Action Buttons */}
                  <div className="flex flex-col sm:flex-row gap-4 justify-center">
                    <button
                      onClick={downloadReport}
                      className="flex items-center justify-center px-6 py-3 bg-indigo-600 hover:bg-indigo-700 text-white font-medium rounded-lg focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 transition-all duration-200"
                    >
                      <Download className="h-4 w-4 mr-2" />
                      Download Report
                    </button>
                    
                    <button
                      onClick={resetAnalysis}
                      className="px-6 py-3 border-2 border-gray-300 text-gray-700 font-medium rounded-lg hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-gray-500 transition-all duration-200"
                    >
                      Analyze Another File
                    </button>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Information Panel */}
        <div className="bg-purple-50 rounded-xl p-6 border border-purple-200">
          <h3 className="text-lg font-semibold text-purple-800 mb-3">
            Detection Technology
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-purple-700">
            <div className="flex items-start">
              <div className="w-2 h-2 bg-purple-500 rounded-full mr-3 mt-2 flex-shrink-0"></div>
              <p>Multi-model ensemble detection system</p>
            </div>
            <div className="flex items-start">
              <div className="w-2 h-2 bg-purple-500 rounded-full mr-3 mt-2 flex-shrink-0"></div>
              <p>Chunk-based analysis for long audio files</p>
            </div>
            <div className="flex items-start">
              <div className="w-2 h-2 bg-purple-500 rounded-full mr-3 mt-2 flex-shrink-0"></div>
              <p>Advanced transformer models for accuracy</p>
            </div>
            <div className="flex items-start">
              <div className="w-2 h-2 bg-purple-500 rounded-full mr-3 mt-2 flex-shrink-0"></div>
              <p>Forensic-grade confidence scoring</p>
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

export default SyntheticDetectionPage;