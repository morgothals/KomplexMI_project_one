import React, { useMemo } from "react";
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
} from "recharts";

const CandleChart = ({ data }) => {
  const chartData = useMemo(() => {
    if (!data || data.length === 0) return [];

    const mapped = data
      .map((candle) => ({
        rawTime: new Date(candle.timestamp).getTime(),
        timeLabel: new Date(candle.timestamp).toLocaleTimeString([], {
          hour: "2-digit",
          minute: "2-digit",
        }),
        price: parseFloat(candle.close),
      }))
      .filter((item) => item.price > 1000 && item.price < 500000);

    const uniqueMap = new Map();
    mapped.forEach((item) => {
      uniqueMap.set(item.rawTime, item);
    });

    return Array.from(uniqueMap.values()).sort((a, b) => a.rawTime - b.rawTime);
  }, [data]);

  if (chartData.length === 0) {
    return (
      <div
        className="card"
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          height: "300px",
          color: "#8b949e",
        }}
      >
        Waiting for data...
      </div>
    );
  }

  const prices = chartData.map((d) => d.price);
  const minPrice = Math.min(...prices);
  const maxPrice = Math.max(...prices);
  const domain = [minPrice - minPrice * 0.005, maxPrice + maxPrice * 0.005];

  const formatDollar = (number) => {
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: "USD",
      maximumFractionDigits: 0,
    }).format(number);
  };

  const formatTime = (timestamp) => {
    return new Date(timestamp).toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  return (
    <div className="card">
      <h3 className="chart-title">Bitcoin Price (1H Close)</h3>
      <div className="chart-container" style={{ height: "300px" }}>
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={chartData}>
            <defs>
              <linearGradient id="colorPrice" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#238636" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#238636" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid
              stroke="#21262d"
              strokeDasharray="3 3"
              vertical={false}
            />

            <XAxis
              dataKey="rawTime"
              type="number"
              domain={["dataMin", "dataMax"]}
              stroke="#8b949e"
              tick={{ fontSize: 11 }}
              tickFormatter={formatTime}
              minTickGap={40}
            />

            <YAxis
              domain={domain}
              stroke="#8b949e"
              tick={{ fontSize: 11 }}
              width={80}
              tickFormatter={formatDollar}
            />

            <Tooltip
              contentStyle={{
                backgroundColor: "#161b22",
                borderColor: "#30363d",
                color: "#e6edf3",
              }}
              itemStyle={{ color: "#238636" }}
              formatter={(value) => [formatDollar(value), "Price"]}
              labelFormatter={(label) => `Time: ${formatTime(label)}`}
            />

            <Area
              type="monotone"
              dataKey="price"
              stroke="#238636"
              strokeWidth={2}
              fillOpacity={1}
              fill="url(#colorPrice)"
              animationDuration={500}
              baseValue="dataMin"
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default CandleChart;
