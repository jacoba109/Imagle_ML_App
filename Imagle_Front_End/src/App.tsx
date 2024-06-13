import React, { useState, ChangeEvent } from 'react';
import axios from 'axios';
import './App.css';

const App: React.FC = () => {
  const [score, setScore] = useState<number | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const handleImageUpload = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('image', file);

    setLoading(true);
    setError(null);

    try {
      const response = await axios.post('http://your-backend-url/score', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setScore(response.data.score);
    } catch (err) {
      setError('Error uploading image');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <h1>Imagle</h1>
      <input type="file" accept="image/*" onChange={handleImageUpload} />
      {loading && <p>Loading...</p>}
      {error && <p style={{ color: 'red' }}>{error}</p>}
      {score !== null && !loading && <p>Score: {score}</p>}
    </div>
  );
};

export default App;

