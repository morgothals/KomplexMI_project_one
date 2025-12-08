import React, { useMemo } from "react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
} from "recharts";

const IntradayChart = ({ data }) => {
  const chartData = useMemo(() => {
    if (!data || data.length === 0) return [];

    const mapped = data
      .map((item) => ({
        rawTime: new Date(item.timestamp).getTime(),
        timeLabel: new Date(item.timestamp).toLocaleTimeString([], {
          hour: "2-digit",
          minute: "2-digit",
        }),
        price: parseFloat(item.close),
      }))
      .filter((item) => item.price > 0);

    const uniqueMap = new Map();
    mapped.forEach((item) => {
      uniqueMap.set(item.rawTime, item);
    });

    return Array.from(uniqueMap.values()).sort((a, b) => a.rawTime - b.rawTime);
  }, [data]);

  if (chartData.length === 0) {
    return (
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          height: "100%",
          color: "#8b949e",
          fontSize: "0.9rem",
        }}
      >
        Waiting for live data...
      </div>
    );
  }

  const prices = chartData.map((d) => d.price);
  const minPrice = Math.min(...prices);
  const maxPrice = Math.max(...prices);
  const domain = [minPrice - minPrice * 0.0005, maxPrice + maxPrice * 0.0005];

  const formatDollar = (number) => {
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: "USD",
      maximumFractionDigits: 0,
    }).format(number);
  };

  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart data={chartData}>
        <CartesianGrid
          stroke="#21262d"
          strokeDasharray="3 3"
          vertical={false}
        />
        <XAxis
          dataKey="timeLabel"
          stroke="#8b949e"
          tick={{ fontSize: 10 }}
          minTickGap={30}
          interval="preserveStartEnd"
        />
        <YAxis
          domain={domain}
          stroke="#8b949e"
          tick={{ fontSize: 10 }}
          width={60}
          tickFormatter={formatDollar}
        />
        <Tooltip
          contentStyle={{
            backgroundColor: "#161b22",
            borderColor: "#30363d",
            color: "#e6edf3",
          }}
          itemStyle={{ color: "#58a6ff" }}
          formatter={(value) => [
            new Intl.NumberFormat("en-US", {
              style: "currency",
              currency: "USD",
            }).format(value),
            "Price",
          ]}
          labelFormatter={(label) => `${label}`}
        />
        <Line
          type="monotone"
          dataKey="price"
          stroke="#58a6ff"
          strokeWidth={2}
          dot={false}
          isAnimationActive={false}
        />
      </LineChart>
    </ResponsiveContainer>
  );
};

export default IntradayChart;
