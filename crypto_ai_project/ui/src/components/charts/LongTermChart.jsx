import React, { useMemo } from "react";
import {
  ResponsiveContainer,
  ComposedChart,
  Area,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from "recharts";

const LongTermChart = ({ data }) => {
  const chartData = useMemo(() => {
    if (!data || !data.labels) return [];

    return data.labels.map((year, index) => ({
      year,
      price: data.pred_price[index],
      low: data.pred_price_low[index],
      high: data.pred_price_high[index],
    }));
  }, [data]);

  if (!data) return <div className="card">Loading Chart...</div>;

  return (
    <div className="card">
      <h3 className="chart-title">
        Long Term Log-Regression Curve (2012-2030)
      </h3>
      <div className="chart-container">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={chartData}>
            <defs>
              <linearGradient id="colorBand" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid
              stroke="#21262d"
              strokeDasharray="3 3"
              vertical={false}
            />
            <XAxis dataKey="year" stroke="#8b949e" tick={{ fontSize: 12 }} />
            <YAxis
              stroke="#8b949e"
              scale="log"
              domain={["auto", "auto"]}
              tick={{ fontSize: 12 }}
              tickFormatter={(val) => `$${val}`}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "#161b22",
                borderColor: "#30363d",
                color: "#e6edf3",
              }}
            />
            <Legend wrapperStyle={{ paddingTop: "10px" }} />

            <Area
              type="monotone"
              dataKey="high"
              stroke="none"
              fill="url(#colorBand)"
            />
            <Line
              type="monotone"
              dataKey="price"
              stroke="#3b82f6"
              strokeWidth={2}
              dot={false}
              name="Predicted Trend"
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default LongTermChart;
