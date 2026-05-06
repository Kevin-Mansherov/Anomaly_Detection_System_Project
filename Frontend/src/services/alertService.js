import api from "./api";

export const alertService = {
  getAllAlerts: async () => {
    try {
      const response = await api.get("/alerts");
      return response.data;
    } catch (error) {
      console.error("Error fetching alerts: ", error);
      throw error;
    }
  },

  updateAlertStatus: async (id, status) => {
    try {
      const response = await api.patch(`/alerts/${id}/status`, { status });
      return response.data;
    } catch (error) {
      console.error(`Error updating alert ${id} status: `, error);
      throw error;
    }
  },
};
