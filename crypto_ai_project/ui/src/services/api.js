import axios from "axios";

const API_BASE_URL = "http://localhost:5000/api";

export const fetchDashboardState = async () => {
  try {
    const response = await axios.get(`${API_BASE_URL}/state`);
    return response.data;
  } catch (error) {
    console.error("API connection failed:", error);
    throw error;
  }
};
