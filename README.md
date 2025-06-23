# Hearing Impaired Sound App

A real-time mobile app that can identify and categorise important environmental sounds, such as sirens and car horns, to help drivers with hearing impairments. To improve road safety, the app transforms these audio cues into visual alerts.
##  Features

- Continuous 2-second audio recording using `react-native-audio-record`
-  Real-time predictions from a remote backend API
-  Color Coded Alerts
  - Red: Hazardous
  - Yellow: Semi-immediate
  - Grey: Neutral
-  Scrollable history of recent predictions (up to 10)

## Tech Stack

- **React Native**
- **TypeScript**

- **Flask**
- Backend endpoint: `https://tgh-hida-bd70b206b3f2.herokuapp.com/predict`

## Setup Instructions

### 1. Clone the Repo

```bash
git clone https://github.com/swethasingireddy/HI_DA.git
cd frontend
cd Hearing_Impaired_Driver_App
```

### 2. Install Dependencies

```bash
npm install
```

Or if you use yarn:

```bash
yarn install
```

### 3. Link Native Modules

```bash
npx react-native link
```

### 4. Run the App

```bash
npx react-native run-android

```

##  Permissions

The app requests the following permissions:

- **RECORD_AUDIO**: To capture sound
- **WRITE_EXTERNAL_STORAGE / READ_MEDIA_AUDIO**: To save and access temporary audio files



## Backend API

The app posts 2-second WAV files to the backend in the format:

```
POST /predict
Content-Type: multipart/form-data
Body:
  - audio: chunk.wav (audio/wav)
```

The backend returns:

```json
{
  "predictions": [
    {
      "class_name": "car_horn",
      "score": 0.95,
      "hazardous": true,
      "semi_immediate": false
    }
  ]
}
```

