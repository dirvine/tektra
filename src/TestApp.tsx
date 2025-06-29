import React from 'react';
import './App.css';

const TestApp: React.FC = () => {
  return (
    <div className="min-h-screen bg-gray-900 text-white p-8">
      <h1 className="text-4xl font-bold text-center">
        Tektra AI Assistant - Test Mode
      </h1>
      <div className="mt-8 text-center">
        <p className="text-lg text-gray-300">
          If you can see this, React and Tailwind are working correctly.
        </p>
        <div className="mt-4 p-4 bg-blue-600 rounded-lg inline-block">
          <p>This is a styled test element</p>
        </div>
      </div>
    </div>
  );
};

export default TestApp;