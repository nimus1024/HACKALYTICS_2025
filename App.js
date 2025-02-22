import React, { useState } from 'react';
import { ReactMic } from 'react-mic';
import axios from 'axios';

function App() {
  const [record, setRecord] = useState(false);
  const [transcription, setTranscription] = useState('');
  const [loading, setLoading] = useState(false);

  const startRecording = () => {
    setRecord(true);
  };

  const stopRecording = () => {
    setRecord(false);
  };

  const onStop = async (recordedBlob) => {
    setLoading(true);
    try {
      const response = await axios.post('YOUR_API_ENDPOINT', recordedBlob, {
        headers: {
          'Content-Type': 'audio/wav',
        },
      });
      setTranscription(response.data.transcription);
    } catch (error) {
      console.error('Error transcribing audio:', error);
    }
    setLoading(false);
  };

  return (
    <div style={styles.container}>
      <h1>Personalized Speech-to-Text Generator</h1>
      <button onClick={record ? stopRecording : startRecording} style={styles.button}>
        {record ? 'Stop Recording' : 'Start Recording'}
      </button>
      <ReactMic
        record={record}
        className="sound-wave"
        onStop={onStop}
        strokeColor="#000000"
        backgroundColor="#FF4081"
      />
      {loading && <p>Loading...</p>}
      <textarea
        value={transcription}
        readOnly
        placeholder="Transcription will appear here..."
        style={styles.textArea}
      />
    </div>
  );
}

const styles = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    padding: '20px',
    fontFamily: 'Arial, sans-serif',
  },
  button: {
    margin: '20px 0',
    padding: '10px 20px',
    fontSize: '16px',
    cursor: 'pointer',
  },
  textArea: {
    width: '100%',
    height: '150px',
    padding: '10px',
    fontSize: '16px',
  },
};

export default App;
