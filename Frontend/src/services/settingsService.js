import api from "./api";

export const settingsService = {
  getSettings: async () => {
    try {
      const response = await api.get("/settings");
      return response.data;
    } catch (error) {
      console.error("Error fetching settings:", error);
      throw error;
    }
  },

  updateSettings: async (packetThresh, flowThresh) => {
    try {
      const currentResponse = await api.get("/settings");
      const currentSettings = currentResponse.data || {};

      const updatedSettings = {
        ...currentSettings,
        packetThreshold: Number(packetThresh),
        flowThreshold: Number(flowThresh),
      };

      const response = await api.put("/settings", updatedSettings);
      return response.data;
    } catch (error) {
      console.error("Error updating settings:", error);
      throw error;
    }
  },
};
