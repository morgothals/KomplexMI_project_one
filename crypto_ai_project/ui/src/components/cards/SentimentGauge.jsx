import React from "react";

const SentimentGauge = ({ value }) => {
  const percentage = Math.min(Math.max(value, 0), 100);

  const getColor = (val) => {
    if (val <= 25) return "#da3633";
    if (val <= 45) return "#d29922";
    if (val <= 55) return "#e3b341";
    if (val <= 75) return "#238636";
    return "#238636";
  };

  const color = getColor(percentage);

  return (
    <div className="gauge-container">
      <div className="gauge-header">
        <span>Fear & Greed </span>
        <span style={{ color: color, fontWeight: "bold", fontSize: "1.2rem" }}>
          {value}
        </span>
      </div>

      <div className="gauge-bar-bg">
        <div
          className="gauge-bar-fill"
          style={{
            width: `${percentage}%`,
            backgroundColor: color,
          }}
        ></div>
      </div>

      <div className="gauge-labels">
        <span>Fear</span>
        <span>Greed</span>
      </div>
    </div>
  );
};

export default SentimentGauge;
