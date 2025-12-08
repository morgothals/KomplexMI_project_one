import React from "react";
import { useCryptoData } from "../hooks/useCryptoData";
import Sidebar from "../components/layout/Sidebar";
import Topbar from "../components/layout/Topbar";
import Loading from "../components/loader/Loading";
import AdvisorMessage from "../components/cards/AdvisorMessage";
import LongTermChart from "../components/charts/LongTermChart";
import CandleChart from "../components/charts/CandleChart";
import IntradayChart from "../components/charts/IntradayChart";
import SentimentGauge from "../components/cards/SentimentGauge";
import StatCard from "../components/cards/StatCard";
import { Newspaper, Zap, Activity } from "lucide-react";

const Dashboard = () => {
  const { data, loading, error } = useCryptoData();

  if (loading) return <Loading text="Connecting to Neural Network..." />;
  if (error)
    return (
      <div className="loader-container" style={{ color: "#da3633" }}>
        Error: Backend Offline
      </div>
    );

  const newsSentiment = data.sentiment?.latest?.news_sentiment || 0;
  const intradayData = data.intraday_1m || [];

  return (
    <div className="dashboard-container">
      <Sidebar />

      <main className="main-content">
        <Topbar status="online" />

        <div className="grid-layout">
          <div className="left-column">
            <AdvisorMessage
              advice={data.advice}
              predictedPrice={data.advice?.predicted_price}
            />

            <LongTermChart data={data.long_curve} />

            <CandleChart data={data.candles_1h} />
          </div>

          <div className="right-column">
            <div className="card">
              <h3 className="chart-title">
                <Activity size={18} /> Market Sentiment
              </h3>

              <SentimentGauge
                value={data.sentiment?.latest?.fear_greed || 50}
              />

              <div
                style={{
                  marginTop: "1.5rem",
                  paddingTop: "1.5rem",
                  borderTop: "1px solid var(--border-color)",
                }}
              >
                <StatCard
                  title="News Sentiment "
                  value={newsSentiment.toFixed(2)}
                  subValue={newsSentiment > 0 ? "Bullish" : "Bearish"}
                  trend={newsSentiment > 0 ? "up" : "down"}
                  icon={Newspaper}
                />
              </div>
            </div>

            <div className="card">
              <h3 className="chart-title">
                <Zap size={18} /> Intraday (1m)
              </h3>
              <div style={{ height: "200px" }}>
                <IntradayChart data={intradayData} />
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
};

export default Dashboard;
