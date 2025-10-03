import axios from 'axios';

export const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:5000/api',
  headers: {
    'Content-Type': 'application/json',
  },
});

// Blockchain service
export const blockchainApi = axios.create({
  baseURL: import.meta.env.VITE_BLOCKCHAIN_URL || 'http://localhost:5001/api',
  headers: {
    'Content-Type': 'application/json',
  },
});

// Python model service
export const modelApi = axios.create({
  baseURL: import.meta.env.VITE_MODEL_URL || 'http://localhost:5002/api',
  headers: {
    'Content-Type': 'application/json',
  },
});