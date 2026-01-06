import React, { useState, FormEvent } from 'react';
import axios from 'axios';
import { useDropzone } from 'react-dropzone';
import './App.css';

const App: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null)
  const [score, setScore] = useState<number>(0);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [corrects, setCorrects] = useState<number>(0);

  const onDrop = (acceptedFiles: File[]) => {
    setSelectedFile(acceptedFiles[0] || null);
    setPreview(URL.createObjectURL(acceptedFiles[0]) || null);
  };

  const handleWin = async () => {

    setLoading(true);
    setError(null);

    try {
      const response = await axios.get('http://127.0.0.1:8000/win', {
        responseType: 'blob'
      });
      const url = URL.createObjectURL(response.data)
      setPreview(url)
    } catch (err) {
      setError('Error retrieving image');
    } finally {
      setLoading(false);
    }
  };

  const handleScore = (score: number) => {
      if (score > 9.55) {
        setCorrects((prevCorrects) => prevCorrects + 1);
        if (corrects + 1 == 3) {
          handleWin();
        }
      }
  };

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault();
    if (!selectedFile) {
      setError('Please select a file before submitting.');
      return;
    }

    const formData = new FormData();
    formData.append('file', selectedFile);

    setLoading(true);
    setError(null);

    try {
      const response = await axios.post('http://127.0.0.1:8000/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setScore(response.data.score);
      handleScore(response.data.score);
    } catch (err) {
      setError('Error uploading image');
    } finally {
      setLoading(false);
    }
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({ onDrop });

  return (
    <div className="App">
      <h1>Imagle</h1>
      <p>Input your picture, and try to get three correct comparisons with the image of the day!</p>
      <form onSubmit={handleSubmit} className="form-container">
        <div {...getRootProps({ className: 'dropzone' })}>
          <input {...getInputProps()} />
          {isDragActive ? (
            <p>Drop the files here ...</p>
          ) : (
            <p>Drag 'n' drop some files here, or click to select files</p>
          )}
          {preview && <img src={preview} alt="Preview" className="preview-image" />}

        </div>
        <button type="submit" disabled={loading || !selectedFile}>
          {loading ? 'Uploading...' : 'Submit'}
        </button>
        {<p style={{color: 'white'}}>Correct attempts: {corrects}</p>}
      </form>
      {error && <p style={{ color: 'red' }}>{error}</p>}
      {score !== null && !loading && <p>Score: {score}</p>}
    </div>
  );
};

export default App;