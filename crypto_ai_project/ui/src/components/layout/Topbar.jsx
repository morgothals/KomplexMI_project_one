import React from "react";
import { Wifi, WifiOff } from "lucide-react";

const Topbar = ({ status }) => {
  return (
    <header className="header">
      <div>
        <h1>Crypto AI Project</h1>
        <p style={{ color: "#8b949e", marginTop: "5px" }}>
          Real-time Bitcoin Strategy & Forecast
        </p>
      </div>

      <div
        className={`status-badge ${
          status === "online" ? "status-online" : "status-offline"
        }`}
      >
        {status === "online" ? <Wifi size={14} /> : <WifiOff size={14} />}
        {status === "online" ? "System Online" : "Backend Offline"}
      </div>
    </header>
  );
};

export default Topbar;
