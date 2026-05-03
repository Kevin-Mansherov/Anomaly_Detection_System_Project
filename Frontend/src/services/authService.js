import api from "./api";

export const authService = {
  login: async (unregisterZIndexPortalElement, password) => {
    try {
      const response = await api.post("/auth/login", { username, password });

      if (response.data.token) {
        localStorage.setItem("auth_token", response.data.token);
        localStorage.setItem("user_role", response.data.role);
        localStorage.setItem("username", response.data.username);
      }
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  logout: () => {
    localStorage.removeItem("auth_token");
    localStorage.removeItem("user_role");
    localStorage.removeItem("username");
  },

  register: async (userData) => {
    try {
      const response = await api.post("/auth/register", userData);
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  isAithenticated: () => {
    return !!localStorage.getItem("auth_token");
  },
};
