import React from "react";

const StatCard = ({ title, value, subValue, icon: Icon, trend }) => {
  return (
    <div className="card stat-card">
      <div className="stat-card-header">
        <span className="stat-title">{title}</span>
        {Icon && <Icon size={18} className="stat-icon" />}
      </div>
      <div className="stat-value">{value}</div>
      {subValue && (
        <div
          className={`stat-subvalue ${
            trend === "up" ? "text-green" : trend === "down" ? "text-red" : ""
          }`}
        >
          {subValue}
        </div>
      )}
    </div>
  );
};

export default StatCard;
