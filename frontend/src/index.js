import React from 'react';
import ReactDOM from 'react-dom';
import './index.css'; // Optional, create this for global styles or remove if not needed
import App from './App'; // Assuming your main component is named App and is in the same folder

// Render the root component
ReactDOM.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
  document.getElementById('root') // Ensure this matches the 'root' div in your index.html
);
