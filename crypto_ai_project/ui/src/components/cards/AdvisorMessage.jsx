import React from "react";
import { Bot, TrendingUp, TrendingDown, Minus } from "lucide-react";

const AdvisorMessage = ({ advice, predictedPrice }) => {
  if (!advice) return null;

  const signalClass =
    advice.signal === "BUY"
      ? "signal-buy"
      : advice.signal === "SELL"
      ? "signal-sell"
      : "signal-hold";

  const SignalIcon =
    advice.signal === "BUY"
      ? TrendingUp
      : advice.signal === "SELL"
      ? TrendingDown
      : Minus;

  return (
    <div className="card advisor-card">
      <div className="bot-avatar">
        <Bot color="white" size={28} />
      </div>

      <div className="message-content">
        <div className="message-header">
          <span style={{ color: "#8b949e", fontSize: "0.9rem" }}>
            AI Advisor • Just now
          </span>
          <span className={`signal-badge ${signalClass}`}>{advice.signal}</span>
        </div>

        <p style={{ lineHeight: "1.6", marginBottom: "1rem" }}>
          Based on current market structure and sentiment analysis, the system
          suggests a <strong className={signalClass}>{advice.signal}</strong>{" "}
          position.
          <br />
          Predicted next closing price:{" "}
          <span className="price-highlight">
            ${predictedPrice ? predictedPrice.toFixed(2) : "---"}
          </span>
        </p>

        <div className="stats-row">
          <div style={{ display: "flex", alignItems: "center", gap: "5px" }}>
            <SignalIcon size={14} />
            Trend: {advice.direction || "Neutral"}
          </div>
          <div>Sentiment Score: {advice.sentiment_score || 0}</div>
        </div>
      </div>
    </div>
  );
};

export default AdvisorMessage;
