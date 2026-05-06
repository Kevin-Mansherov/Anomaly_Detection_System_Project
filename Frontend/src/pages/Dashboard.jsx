import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import {
  ShieldAlert,
  LogOut,
  Activity,
  Server,
  Clock,
  AlertTriangle,
  Search,
  Filter,
  CheckCircle,
  RefreshCw,
  Settings as SettingsIcon,
} from "lucide-react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import { authService } from "../services/authService";
import { alertService } from "../services/alertService";

const Dashboard = () => {
  const navigate = useNavigate();
  const username = localStorage.getItem("username");

  const [alerts, setAlerts] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState("");

  // States לסינון וחיפוש
  const [searchTerm, setSearchTerm] = useState("");
  const [severityFilter, setSeverityFilter] = useState("ALL");

  // State לרענון אוטומטי (זמן אמת)
  const [isAutoRefresh, setIsAutoRefresh] = useState(true);

  const fetchAlerts = async (showLoading = true) => {
    try {
      if (showLoading) setIsLoading(true);
      const data = await alertService.getAllAlerts();
      const sortedData = data.sort(
        (a, b) => new Date(b.timestamp) - new Date(a.timestamp),
      );
      setAlerts(sortedData);
      setError("");
    } catch (err) {
      console.error("Failed to fetch alerts:", err);
      setError("Failed to load alerts.");
      if (
        err.response &&
        (err.response.status === 401 || err.response.status === 403)
      ) {
        authService.logout();
        navigate("/login");
      }
    } finally {
      if (showLoading) setIsLoading(false);
    }
  };

  // אפקט למשיכה ראשונית ורענון אוטומטי (Polling)
  useEffect(() => {
    fetchAlerts();

    let interval;
    if (isAutoRefresh) {
      // רענון ברקע כל 3 שניות בלי להראות את ספינר הטעינה
      interval = setInterval(() => fetchAlerts(false), 3000);
    }

    return () => {
      if (interval) clearInterval(interval);
    };
  }, [isAutoRefresh]);

  const handleLogout = () => {
    authService.logout();
    navigate("/login");
  };

  // פעולת שינוי סטטוס
  const handleResolveAlert = async (id) => {
    try {
      await alertService.updateAlertStatus(id, "RESOLVED");
      // עדכון מקומי מהיר לתחושה חלקה, השרת יתעדכן ברקע
      setAlerts(
        alerts.map((a) => (a.id === id ? { ...a, status: "RESOLVED" } : a)),
      );
    } catch (err) {
      console.error("Failed to update status", err);
    }
  };

  // לוגיקת הסינון
  const filteredAlerts = alerts.filter((alert) => {
    const matchesSearch =
      alert.sourceIp.includes(searchTerm) ||
      alert.destinationIp.includes(searchTerm);
    const matchesSeverity =
      severityFilter === "ALL" || alert.severity === severityFilter;
    return matchesSearch && matchesSeverity;
  });

  // הכנת נתונים לגרף - הבלוק שהיה חסר!
  const severityCounts = { CRITICAL: 0, HIGH: 0, MEDIUM: 0 };
  filteredAlerts.forEach((a) => {
    if (severityCounts[a.severity] !== undefined) severityCounts[a.severity]++;
  });

  const chartData = [
    { name: "CRITICAL", count: severityCounts.CRITICAL, color: "#ef4444" }, // Red
    { name: "HIGH", count: severityCounts.HIGH, color: "#f97316" }, // Orange
    { name: "MEDIUM", count: severityCounts.MEDIUM, color: "#eab308" }, // Yellow
  ];

  const getSeverityBadge = (severity) => {
    switch (severity?.toUpperCase()) {
      case "CRITICAL":
        return (
          <span className="px-2 py-1 bg-red-900/50 text-red-400 border border-red-500/50 rounded text-xs font-bold uppercase">
            Critical
          </span>
        );
      case "HIGH":
        return (
          <span className="px-2 py-1 bg-orange-900/50 text-orange-400 border border-orange-500/50 rounded text-xs font-bold uppercase">
            High
          </span>
        );
      case "MEDIUM":
        return (
          <span className="px-2 py-1 bg-yellow-900/50 text-yellow-400 border border-yellow-500/50 rounded text-xs font-bold uppercase">
            Medium
          </span>
        );
      default:
        return (
          <span className="px-2 py-1 bg-slate-800 text-slate-300 border border-slate-600 rounded text-xs font-bold uppercase">
            {severity || "UNKNOWN"}
          </span>
        );
    }
  };

  const formatDate = (dateString) => {
    if (!dateString) return "N/A";
    return new Date(dateString).toLocaleString("en-GB");
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-300 font-sans pb-10">
      <header className="bg-slate-900 border-b border-slate-800 sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center gap-3">
              <ShieldAlert className="h-8 w-8 text-emerald-500" />
              <div>
                <h1 className="text-xl font-bold text-white tracking-tight">
                  IDS Control Center
                </h1>
                <p className="text-xs text-slate-500">
                  Real-Time Anomaly Detection
                </p>
              </div>
            </div>
            <div className="flex items-center gap-6">
              <div className="text-sm">
                <span className="text-slate-500">Analyst: </span>
                <span className="text-emerald-400 font-medium">{username}</span>
              </div>
              <button
                onClick={handleLogout}
                className="flex items-center gap-2 px-4 py-2 bg-slate-800 hover:bg-slate-700 text-white rounded-lg transition-colors text-sm border border-slate-700"
              >
                <LogOut className="h-4 w-4" /> Logout
              </button>
              <button
                onClick={() => navigate("/settings")}
                className="flex items-center gap-2 px-4 py-2 bg-slate-800 hover:bg-slate-700 text-white rounded-lg transition-colors text-sm border border-slate-700 mr-2"
              >
                <SettingsIcon className="h-4 w-4" /> Settings
              </button>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Stats & Graph Section */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
          {/* Stats Column */}
          <div className="flex flex-col gap-6">
            <div className="bg-slate-900 p-6 rounded-xl border border-slate-800 flex items-center gap-4">
              <div className="p-3 bg-blue-900/30 text-blue-500 rounded-lg">
                <Activity className="h-6 w-6" />
              </div>
              <div>
                <p className="text-sm text-slate-400 font-medium">
                  Filtered Alerts
                </p>
                <h3 className="text-2xl font-bold text-white">
                  {filteredAlerts.length}
                </h3>
              </div>
            </div>
            <div className="bg-slate-900 p-6 rounded-xl border border-slate-800 flex items-center gap-4">
              <div className="p-3 bg-red-900/30 text-red-500 rounded-lg">
                <AlertTriangle className="h-6 w-6" />
              </div>
              <div>
                <p className="text-sm text-slate-400 font-medium">
                  Critical Threats
                </p>
                <h3 className="text-2xl font-bold text-white">
                  {severityCounts.CRITICAL}
                </h3>
              </div>
            </div>
          </div>

          {/* Graph Column */}
          <div className="lg:col-span-2 bg-slate-900 p-6 rounded-xl border border-slate-800">
            <h3 className="text-sm font-bold text-slate-400 mb-4">
              Threat Distribution
            </h3>
            <div className="h-48">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={chartData}
                  margin={{ top: 0, right: 0, left: -20, bottom: 0 }}
                >
                  <XAxis
                    dataKey="name"
                    stroke="#64748b"
                    fontSize={12}
                    tickLine={false}
                    axisLine={false}
                  />
                  <YAxis
                    stroke="#64748b"
                    fontSize={12}
                    tickLine={false}
                    axisLine={false}
                  />
                  <Tooltip
                    cursor={{ fill: "#1e293b" }}
                    contentStyle={{
                      backgroundColor: "#0f172a",
                      borderColor: "#1e293b",
                      color: "#fff",
                    }}
                  />
                  <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                    {chartData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        {/* Toolbar: Search, Filter, Auto-Refresh */}
        <div className="flex flex-col sm:flex-row justify-between items-center gap-4 mb-6 bg-slate-900 p-4 rounded-xl border border-slate-800">
          <div className="flex flex-1 items-center gap-4 w-full">
            <div className="relative flex-1 max-w-md">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-500" />
              <input
                type="text"
                placeholder="Search by IP address..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full bg-slate-950 border border-slate-700 text-sm rounded-lg pl-10 pr-4 py-2 text-white focus:outline-none focus:border-emerald-500 transition-colors"
              />
            </div>
            <div className="flex items-center gap-2 bg-slate-950 border border-slate-700 rounded-lg px-3 py-2">
              <Filter className="h-4 w-4 text-slate-500" />
              <select
                value={severityFilter}
                onChange={(e) => setSeverityFilter(e.target.value)}
                className="bg-transparent text-sm text-white focus:outline-none cursor-pointer"
              >
                <option value="ALL">All Severities</option>
                <option value="CRITICAL">Critical Only</option>
                <option value="HIGH">High Only</option>
                <option value="MEDIUM">Medium Only</option>
              </select>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <button
              onClick={() => setIsAutoRefresh(!isAutoRefresh)}
              className={`flex items-center gap-2 px-3 py-2 text-sm rounded-lg border transition-colors ${isAutoRefresh ? "bg-emerald-900/30 border-emerald-500/50 text-emerald-400" : "bg-slate-800 border-slate-700 text-slate-400 hover:text-white"}`}
            >
              <RefreshCw
                className={`h-4 w-4 ${isAutoRefresh ? "animate-spin-slow" : ""}`}
              />
              {isAutoRefresh ? "Live Sync ON" : "Live Sync OFF"}
            </button>
          </div>
        </div>

        {/* Alerts Table */}
        <div className="bg-slate-900 rounded-xl border border-slate-800 shadow-xl overflow-hidden">
          <div className="p-4 border-b border-slate-800 flex items-center bg-slate-900/50">
            <h2 className="text-lg font-bold text-white flex items-center gap-2">
              <Clock className="h-5 w-5 text-slate-400" /> Event Logs
            </h2>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="bg-slate-950/50 border-b border-slate-800 text-xs uppercase tracking-wider text-slate-400">
                  <th className="p-4 font-medium">Timestamp</th>
                  <th className="p-4 font-medium">Source IP</th>
                  <th className="p-4 font-medium">Target IP</th>
                  <th className="p-4 font-medium">Score</th>
                  <th className="p-4 font-medium">Severity</th>
                  <th className="p-4 font-medium">Status</th>
                  <th className="p-4 font-medium text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/50">
                {filteredAlerts.length === 0 ? (
                  <tr>
                    <td colSpan="7" className="p-8 text-center text-slate-500">
                      No alerts match your search.
                    </td>
                  </tr>
                ) : (
                  filteredAlerts.map((alert) => (
                    <tr
                      key={alert.id}
                      className={`hover:bg-slate-800/30 transition-colors ${alert.status === "RESOLVED" ? "opacity-50" : ""}`}
                    >
                      <td className="p-4 text-sm font-mono text-slate-300">
                        {formatDate(alert.timestamp)}
                      </td>
                      <td className="p-4 text-sm font-mono">
                        {alert.sourceIp}
                      </td>
                      <td className="p-4 text-sm font-mono">
                        {alert.destinationIp}
                      </td>
                      <td className="p-4 text-sm font-mono text-slate-400">
                        {alert.anomalyScore
                          ? alert.anomalyScore.toFixed(2)
                          : "-"}
                      </td>
                      <td className="p-4">
                        {getSeverityBadge(alert.severity)}
                      </td>
                      <td className="p-4">
                        <span className="flex items-center gap-1.5 text-sm">
                          <span
                            className={`w-2 h-2 rounded-full ${alert.status === "OPEN" ? "bg-red-500 animate-pulse" : "bg-emerald-500"}`}
                          ></span>
                          {alert.status}
                        </span>
                      </td>
                      <td className="p-4 text-right">
                        {alert.status === "OPEN" && (
                          <button
                            onClick={() => handleResolveAlert(alert.id)}
                            className="text-xs flex items-center gap-1 px-2 py-1 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded border border-slate-600 transition-colors ml-auto"
                          >
                            <CheckCircle className="h-3 w-3 text-emerald-400" />{" "}
                            Resolve
                          </button>
                        )}
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>
      </main>
    </div>
  );
};

export default Dashboard;
