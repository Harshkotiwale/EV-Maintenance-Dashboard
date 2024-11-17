// src/components/ui/alert.js

import React from 'react';
// import './alert.css';

const Alert = ({ message, type = 'info' }) => {
  let alertClass = '';

  switch (type) {
    case 'success':
      alertClass = 'alert-success';
      break;
    case 'error':
      alertClass = 'alert-error';
      break;
    case 'warning':
      alertClass = 'alert-warning';
      break;
    default:
      alertClass = 'alert-info';
  }

  return (
    <div className={`alert ${alertClass}`}>
      <p>{message}</p>
    </div>
  );
};

export const AlertTitle = ({ title }) => {
  return <h4 className="alert-title">{title}</h4>;
};

export const AlertDescription = ({ description }) => {
  return <p className="alert-description">{description}</p>;
};

export { Alert };  

{/* <div className="background-animation"></div> */}
