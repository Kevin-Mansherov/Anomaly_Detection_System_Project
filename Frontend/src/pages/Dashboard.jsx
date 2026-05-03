import React from "react";
import { useNavigate } from "react-router-dom";
import { authService } from "../services/authService";

const Dashboard = () => {
  const navigate = useNavigate();
  const username = localStorage.getItem("username");

  const handleLogout = () => {
    authService.logout();
    navigate("/login");
  };

  return (
    <div className="min-h-screen bg-slate-900 text-white flex flex-col items-center justify-center">
      <h1 className="text-4xl font-bold text-emerald-400 mb-4">
        Welcome to IDS Dashboard
      </h1>
      <p className="text-lg mb-8">
        Logged in as:{" "}
        <span className="text-blue-400 font-semibold">{username}</span>
      </p>
      <button
        onClick={handleLogout}
        className="px-6 py-2 bg-red-600 hover:bg-red-700 rounded-md font-medium transition-colors"
      >
        Logout
      </button>
    </div>
  );
};

export default Dashboard;
