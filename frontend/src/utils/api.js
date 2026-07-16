import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || `http://${window.location.hostname}:5000`;

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60000,
});

export default api;