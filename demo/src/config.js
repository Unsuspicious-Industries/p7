// API configuration
// Runtime-configurable via public/config.js for deployments on different servers.
// Fallbacks: REACT_APP_API_URL -> localhost.
const runtimeConfig = typeof window !== 'undefined' ? window.__P7_CONFIG__ : undefined;
export const API_BASE_URL =
  (runtimeConfig && runtimeConfig.API_BASE_URL) ||
  process.env.REACT_APP_API_URL ||
  'http://localhost:5001';
