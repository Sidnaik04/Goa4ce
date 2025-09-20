#!/usr/bin/env python3
"""
Demo script for Voice Biometric Recognition System
Creates sample data and demonstrates API usage
"""

import requests
import json
import os
import time
import numpy as np
import wave
from pathlib import Path

BASE_URL = "http://localhost:8000"

def create_sample_audio_file(filename, duration=3):
    """Create a simple audio file for testing"""
    try:
        import numpy as np
        import wave
        
        # Generate simple sine wave
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        frequency = 440 + hash(filename) % 200  # Vary frequency based on filename
        wave_data = np.sin(frequency * 2 * np.pi * t) * 0.3
        
        # Convert to 16-bit integers
        audio_data = (wave_data * 32767).astype(np.int16)
        
        # Save as WAV file
        with wave.open(filename, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 2 bytes per sample
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())
            
        print(f"Created sample audio file: {filename}")
        return True
    except ImportError:
        print("numpy and/or wave module not available. Please create audio files manually.")
        return False

def login():
    """Login and get access token"""
    print("🔐 Logging in...")
    
    response = requests.post(
        f"{BASE_URL}/auth/login",
        json={"username": "officer", "password": "police123"}
    )
    
    if response.status_code == 200:
        token = response.json()["access_token"]
        print("✅ Login successful!")
        return token
    else:
        print(f"❌ Login failed: {response.text}")
        return None

def create_inmate(token, name, inmate_code, audio_file):
    """Create a new inmate with reference audio"""
    print(f"👤 Creating inmate: {name} ({inmate_code})")
    
    headers = {"Authorization": f"Bearer {token}"}
    
    with open(audio_file, 'rb') as f:
        files = {'reference_audio': f}
        data = {
            'name': name,
            'inmate_code': inmate_code
        }
        
        response = requests.post(
            f"{BASE_URL}/inmates",
            headers=headers,
            files=files,
            data=data
        )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Inmate created successfully! ID: {result['id']}")
        return result
    else:
        print(f"❌ Failed to create inmate: {response.text}")
        return None

def analyze_audio(token, audio_file, inmate_code=None, claimed_caller="Unknown", context="Demo analysis"):
    """Upload and analyze audio file"""
    print(f"🔍 Analyzing audio file: {audio_file}")
    
    headers = {"Authorization": f"Bearer {token}"}
    
    with open(audio_file, 'rb') as f:
        files = {'audio': f}
        data = {
            'claimed_caller': claimed_caller,
            'context': context,
            'provided_by': 'Demo Script'
        }
        
        if inmate_code:
            data['inmate_code'] = inmate_code
        
        response = requests.post(
            f"{BASE_URL}/upload",
            headers=headers,
            files=files,
            data=data
        )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Analysis completed!")
        print(f"   Report ID: {result['report_code']}")
        print(f"   Synthetic: {'Yes' if result['is_synthetic'] else 'No'} ({result['synthetic_confidence']:.2%})")
        print(f"   Speaker Match: {'Yes' if result['speaker_match'] else 'No'} ({result['speaker_match_confidence']:.2%})")
        if result['matched_inmate_code']:
            print(f"   Matched Inmate: {result['matched_inmate_code']}")
        return result
    else:
        print(f"❌ Analysis failed: {response.text}")
        return None

def get_reports(token):
    """Get all reports"""
    print("📊 Fetching reports...")
    
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(f"{BASE_URL}/reports", headers=headers)
    
    if response.status_code == 200:
        reports = response.json()
        print(f"✅ Found {len(reports)} reports")
        for report in reports:
            print(f"   - {report['report_code']}: {'Synthetic' if report['is_synthetic'] else 'Authentic'}")
        return reports
    else:
        print(f"❌ Failed to fetch reports: {response.text}")
        return []

def download_report(token, report_id, report_code):
    """Download PDF report"""
    print(f"📄 Downloading PDF report: {report_code}")
    
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(f"{BASE_URL}/reports/{report_id}/pdf", headers=headers)
    
    if response.status_code == 200:
        filename = f"forensic_report_{report_code}.pdf"
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"✅ PDF saved as: {filename}")
        return filename
    else:
        print(f"❌ Failed to download PDF: {response.text}")
        return None

def verify_report(token, report_id):
    """Verify report signature"""
    print(f"🔐 Verifying report signature...")
    
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(f"{BASE_URL}/reports/{report_id}/verify", headers=headers)
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Verification result: {'Valid' if result['is_valid'] else 'Invalid'}")
        print(f"   Message: {result['message']}")
        return result['is_valid']
    else:
        print(f"❌ Verification failed: {response.text}")
        return False

def run_demo():
    """Run complete demo scenario"""
    print("🚀 Starting Voice Biometric Recognition Demo")
    print("=" * 50)
    
    # Create sample audio files
    os.makedirs("demo_audio", exist_ok=True)
    
    audio_files = [
        "demo_audio/john_reference.wav",
        "demo_audio/john_test.wav",
        "demo_audio/jane_test.wav",
        "demo_audio/synthetic_test.wav"
    ]
    
    for audio_file in audio_files:
        if not os.path.exists(audio_file):
            create_sample_audio_file(audio_file)
    
    # Step 1: Login
    token = login()
    if not token:
        return
    
    time.sleep(1)
    
    # Step 2: Create inmates
    inmate1 = create_inmate(token, "John Doe", "INM001", audio_files[0])
    if not inmate1:
        return
    
    time.sleep(1)
    
    # Step 3: Analyze audio samples
    print("\n" + "=" * 50)
    
    # Test 1: Same speaker (should match)
    result1 = analyze_audio(
        token, 
        audio_files[1], 
        inmate_code="INM001",
        claimed_caller="John Doe",
        context="Phone call verification - same speaker test"
    )
    
    time.sleep(1)
    
    # Test 2: Different speaker (should not match)
    result2 = analyze_audio(
        token,
        audio_files[2],
        inmate_code="INM001", 
        claimed_caller="Jane Smith",
        context="Phone call verification - different speaker test"
    )
    
    time.sleep(1)
    
    # Test 3: Unknown caller (database search)
    result3 = analyze_audio(
        token,
        audio_files[3],
        claimed_caller="Unknown Caller",
        context="Suspicious call - synthetic voice test"
    )
    
    time.sleep(1)
    
    # Step 4: Get all reports
    print("\n" + "=" * 50)
    reports = get_reports(token)
    
    # Step 5: Download and verify first report
    if reports and result1:
        print("\n" + "=" * 50)
        pdf_file = download_report(token, result1['report_id'], result1['report_code'])
        
        if pdf_file:
            verify_report(token, result1['report_id'])
    
    print("\n" + "=" * 50)
    print("🎉 Demo completed successfully!")
    print("\nDemo Summary:")
    print("- Created sample inmate with reference voice")
    print("- Tested voice matching with different scenarios")
    print("- Generated forensic reports with digital signatures")
    print("- Verified report authenticity")
    print("\nCheck the generated PDF reports and visit http://localhost:8000/docs for API documentation.")

if __name__ == "__main__":
    try:
        run_demo()
    except KeyboardInterrupt:
        print("\n❌ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()