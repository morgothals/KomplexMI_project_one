import React from "react";

const Loading = ({ text }) => {
  return (
    <div className="loader-container">
      <div className="spinner"></div>
      <p style={{ marginTop: "1rem", color: "#8b949e" }}>
        {text || "Loading..."}
      </p>
    </div>
  );
};

export default Loading;
