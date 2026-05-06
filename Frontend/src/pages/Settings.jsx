import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom"; // ייבוא בשביל כפתור החזרה
import { settingsService } from "../services/settingsService";

const Settings = () => {
  const navigate = useNavigate(); // הפעלת הניווט
  const [packetThreshold, setPacketThreshold] = useState("");
  const [flowThreshold, setFlowThreshold] = useState("");
  const [message, setMessage] = useState(null);

  useEffect(() => {
    const fetchSettings = async () => {
      try {
        const data = await settingsService.getSettings();
        console.log("Settings loaded from server:", data); // הדפסה לבדיקה
        setPacketThreshold(
          data.packetThreshold !== undefined ? data.packetThreshold : "",
        );
        setFlowThreshold(
          data.flowThreshold !== undefined ? data.flowThreshold : "",
        );
      } catch (error) {
        console.error("Error loading settings:", error);
        setMessage({ type: "error", text: "Failed to load current settings." });
      }
    };
    fetchSettings();
  }, []);

  const handleSave = async () => {
    console.log(
      "Attempting to save thresholds:",
      packetThreshold,
      flowThreshold,
    );
    try {
      await settingsService.updateSettings(packetThreshold, flowThreshold);
      console.log("Settings successfully saved to DB!");
      setMessage({ type: "success", text: "Settings updated successfully!" });
      setTimeout(() => setMessage(null), 3000);
    } catch (error) {
      console.error("Failed to save. Error details:", error);
      setMessage({
        type: "error",
        text: "Failed to save settings. Check console (F12).",
      });
    }
  };

  return (
    <div className="p-6">
      {/* כפתור חזרה לדשבורד */}
      <button
        onClick={() => navigate("/dashboard")}
        className="mb-6 flex items-center text-gray-400 hover:text-white transition duration-200"
      >
        <svg
          className="w-5 h-5 mr-2"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth="2"
            d="M10 19l-7-7m0 0l7-7m-7 7h18"
          ></path>
        </svg>
        Back to Dashboard
      </button>

      <h2 className="text-2xl font-bold mb-6 text-white">
        Detection Model Calibration
      </h2>

      <div className="bg-[#1e293b] p-6 rounded-lg max-w-xl border border-gray-700">
        <p className="text-gray-400 mb-6 text-sm">
          Adjust the Mean Squared Error (MSE) thresholds for both models
          independently.
        </p>

        <div className="mb-6">
          <label className="block text-sm font-medium mb-2 text-gray-200">
            Packet Model Threshold
          </label>
          <input
            type="number"
            step="0.000001"
            value={packetThreshold}
            onChange={(e) => setPacketThreshold(e.target.value)}
            className="w-full bg-[#0f172a] border border-gray-600 rounded p-3 text-white focus:outline-none focus:border-blue-500"
          />
        </div>

        <div className="mb-8">
          <label className="block text-sm font-medium mb-2 text-gray-200">
            Flow Model Threshold
          </label>
          <input
            type="number"
            step="0.000001"
            value={flowThreshold}
            onChange={(e) => setFlowThreshold(e.target.value)}
            className="w-full bg-[#0f172a] border border-gray-600 rounded p-3 text-white focus:outline-none focus:border-blue-500"
          />
        </div>

        {message && (
          <div
            className={`p-4 mb-6 rounded flex items-center ${message.type === "success" ? "bg-green-900/50 text-green-400 border border-green-800" : "bg-red-900/50 text-red-400 border border-red-800"}`}
          >
            {message.text}
          </div>
        )}

        <button
          onClick={handleSave}
          className="bg-emerald-600 hover:bg-emerald-500 text-white font-medium py-2 px-6 rounded transition duration-200"
        >
          Save Configuration
        </button>
      </div>
    </div>
  );
};

export default Settings;
