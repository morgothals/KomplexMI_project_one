import { useState, useEffect } from "react";
import { fetchDashboardState } from "../services/api";

export const useCryptoData = () => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const loadData = async () => {
      try {
        const result = await fetchDashboardState();
        setData(result);
        setError(null);
      } catch (err) {
        setError(err);
      } finally {
        setLoading(false);
      }
    };

    loadData();

    const intervalId = setInterval(loadData, 60000);

    return () => clearInterval(intervalId);
  }, []);

  return { data, loading, error };
};
