import React from "react";
import {
  LayoutDashboard,
  TrendingUp,
  Settings,
  HelpCircle,
} from "lucide-react";

const Sidebar = () => {
  return (
    <div className="sidebar">
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "10px",
          fontWeight: "bold",
          fontSize: "1.2rem",
        }}
      >
        <div
          style={{
            width: "24px",
            height: "24px",
            background: "#3b82f6",
            borderRadius: "4px",
          }}
        ></div>
        CryptoAI
      </div>

      <ul className="sidebar-menu">
        <li className="sidebar-item active">
          <LayoutDashboard size={18} />
          Dashboard
        </li>
      </ul>
    </div>
  );
};

export default Sidebar;
