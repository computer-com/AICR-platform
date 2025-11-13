"use client"

import { useState, useEffect } from "react"
import { IncidentMap } from "@/components/incident-map"
import { ReportForm } from "@/components/report-form"
import { StatsChart } from "@/components/stats-chart"

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"

interface Incident {
  id: number
  description: string
  latitude: number
  longitude: number
  crisis_type: string
  confidence: number
  status: string
  timestamp: string
  source: string
}

interface Stats {
  total_incidents: number
  active_incidents: number
  resolved_incidents: number
  avg_confidence: number
}

const crisisEmojis: Record<string, string> = {
  flood: "🌊",
  fire: "🔥",
  earthquake: "⚠️",
  gas_leak: "☢️",
  landslide: "🏔️",
  power_outage: "⚡",
  storm: "🌪️",
}

const crisisColors: Record<string, string> = {
  flood: "bg-blue-500",
  fire: "bg-red-500",
  earthquake: "bg-purple-500",
  gas_leak: "bg-yellow-500",
  landslide: "bg-amber-700",
  power_outage: "bg-gray-500",
  storm: "bg-emerald-500",
}

export default function Dashboard() {
  const [incidents, setIncidents] = useState<Incident[]>([])
  const [stats, setStats] = useState<Stats | null>(null)
  const [loading, setLoading] = useState(true)
  const [selectedFilter, setSelectedFilter] = useState("all")
  const [searchQuery, setSearchQuery] = useState("")
  const [showReportForm, setShowReportForm] = useState(false)
  const [activeTab, setActiveTab] = useState("map")

  const fetchIncidents = async () => {
    try {
      const response = await fetch(`${API_URL}/incidents`)
      const data = await response.json()
      setIncidents(data)
    } catch (error) {
      console.error("Error fetching incidents:", error)
    }
  }

  const fetchStats = async () => {
    try {
      const response = await fetch(`${API_URL}/stats`)
      const data = await response.json()
      setStats(data)
    } catch (error) {
      console.error("Error fetching stats:", error)
    }
  }

  useEffect(() => {
    const loadData = async () => {
      setLoading(true)
      await Promise.all([fetchIncidents(), fetchStats()])
      setLoading(false)
    }
    loadData()

    const interval = setInterval(() => {
      fetchIncidents()
      fetchStats()
    }, 10000)

    return () => clearInterval(interval)
  }, [])

  const filteredIncidents = incidents.filter((inc) => {
    const matchesFilter = selectedFilter === "all" || inc.crisis_type === selectedFilter
    const matchesSearch =
      searchQuery === "" ||
      inc.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
      inc.crisis_type.toLowerCase().includes(searchQuery.toLowerCase())
    return matchesFilter && matchesSearch
  })

  const crisisTypes = Array.from(new Set(incidents.map((inc) => inc.crisis_type)))

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-slate-950">
        <div className="text-center space-y-4">
          <div className="animate-pulse text-4xl">⚡</div>
          <p className="text-xl font-semibold text-white">Loading Crisis Command Center...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-slate-950 text-white">
      {/* Header */}
      <header className="sticky top-0 z-50 border-b border-slate-800 bg-slate-900/95 backdrop-blur">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="flex items-center justify-center h-10 w-10 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600">
                <span className="text-2xl">🚨</span>
              </div>
              <div>
                <h1 className="text-2xl font-bold">AICR Crisis Command</h1>
                <p className="text-sm text-slate-400">AI-Powered Emergency Response</p>
              </div>
            </div>
            <button
              onClick={() => setShowReportForm(true)}
              className="px-6 py-2 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg font-semibold hover:opacity-90"
            >
              + Report Incident
            </button>
          </div>
        </div>

        {/* Stats Bar */}
        {stats && (
          <div className="border-t border-slate-800 bg-slate-900/50">
            <div className="container mx-auto px-6 py-3">
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="flex items-center gap-3">
                  <div className="h-10 w-10 rounded-lg bg-blue-500/10 flex items-center justify-center">
                    <span className="text-xl">📊</span>
                  </div>
                  <div>
                    <p className="text-xs text-slate-400">Total</p>
                    <p className="text-lg font-bold">{stats.total_incidents}</p>
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  <div className="h-10 w-10 rounded-lg bg-red-500/10 flex items-center justify-center">
                    <span className="text-xl">⚠️</span>
                  </div>
                  <div>
                    <p className="text-xs text-slate-400">Active</p>
                    <p className="text-lg font-bold">{stats.active_incidents}</p>
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  <div className="h-10 w-10 rounded-lg bg-green-500/10 flex items-center justify-center">
                    <span className="text-xl">✅</span>
                  </div>
                  <div>
                    <p className="text-xs text-slate-400">Resolved</p>
                    <p className="text-lg font-bold">{stats.resolved_incidents}</p>
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  <div className="h-10 w-10 rounded-lg bg-purple-500/10 flex items-center justify-center">
                    <span className="text-xl">🎯</span>
                  </div>
                  <div>
                    <p className="text-xs text-slate-400">Confidence</p>
                    <p className="text-lg font-bold">{(stats.avg_confidence * 100).toFixed(1)}%</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-6 py-6">
        {/* Tabs */}
        <div className="flex gap-2 mb-6">
          <button
            onClick={() => setActiveTab("map")}
            className={`px-6 py-2 rounded-lg font-medium transition ${
              activeTab === "map"
                ? "bg-blue-500 text-white"
                : "bg-slate-800 text-slate-300 hover:bg-slate-700"
            }`}
          >
            🗺️ Live Map
          </button>
          <button
            onClick={() => setActiveTab("list")}
            className={`px-6 py-2 rounded-lg font-medium transition ${
              activeTab === "list"
                ? "bg-blue-500 text-white"
                : "bg-slate-800 text-slate-300 hover:bg-slate-700"
            }`}
          >
            📋 Incidents List
          </button>
        </div>

        {/* Map Tab */}
        {activeTab === "map" && (
          <div className="bg-slate-900 rounded-lg border border-slate-800 overflow-hidden">
            <div className="p-4 border-b border-slate-800">
              <h2 className="text-xl font-bold">Live Crisis Map</h2>
              <p className="text-sm text-slate-400">Real-time geographic distribution</p>
            </div>
            <IncidentMap incidents={filteredIncidents} />
          </div>
        )}

        {/* List Tab */}
        {activeTab === "list" && (
          <div className="grid gap-6 lg:grid-cols-3">
            {/* Incidents List */}
            <div className="lg:col-span-2 bg-slate-900 rounded-lg border border-slate-800 p-6">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-bold">Recent Incidents</h2>
                <input
                  type="text"
                  placeholder="Search..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="px-4 py-2 bg-slate-800 rounded-lg border border-slate-700 focus:border-blue-500 focus:outline-none"
                />
              </div>

              {/* Filters */}
              <div className="flex flex-wrap gap-2 mb-4">
                <button
                  onClick={() => setSelectedFilter("all")}
                  className={`px-3 py-1 rounded-full text-sm ${
                    selectedFilter === "all"
                      ? "bg-blue-500 text-white"
                      : "bg-slate-800 text-slate-300 hover:bg-slate-700"
                  }`}
                >
                  All ({incidents.length})
                </button>
                {crisisTypes.map((type) => (
                  <button
                    key={type}
                    onClick={() => setSelectedFilter(type)}
                    className={`px-3 py-1 rounded-full text-sm flex items-center gap-1 ${
                      selectedFilter === type
                        ? "bg-blue-500 text-white"
                        : "bg-slate-800 text-slate-300 hover:bg-slate-700"
                    }`}
                  >
                    <span>{crisisEmojis[type] || "⚠️"}</span>
                    {type.replace("_", " ")} ({incidents.filter((i) => i.crisis_type === type).length})
                  </button>
                ))}
              </div>

              {/* Incidents */}
              <div className="space-y-3 max-h-[600px] overflow-y-auto">
                {filteredIncidents.length === 0 ? (
                  <div className="text-center py-8 text-slate-400">No incidents found</div>
                ) : (
                  filteredIncidents.map((incident) => (
                    <div
                      key={incident.id}
                      className="bg-slate-800 rounded-lg p-4 hover:bg-slate-750 transition"
                    >
                      <div className="flex items-start justify-between mb-2">
                        <div className="flex items-center gap-3">
                          <div className={`h-10 w-10 rounded-lg ${crisisColors[incident.crisis_type]} flex items-center justify-center text-xl`}>
                            {crisisEmojis[incident.crisis_type] || "⚠️"}
                          </div>
                          <div>
                            <h3 className="font-semibold capitalize">
                              {incident.crisis_type.replace("_", " ")}
                            </h3>
                            <p className="text-xs text-slate-400">
                              {new Date(incident.timestamp).toLocaleString()}
                            </p>
                          </div>
                        </div>
                        <span
                          className={`px-2 py-1 rounded text-xs font-semibold ${
                            incident.status === "active"
                              ? "bg-red-500/20 text-red-400"
                              : "bg-green-500/20 text-green-400"
                          }`}
                        >
                          {incident.status}
                        </span>
                      </div>
                      <p className="text-sm text-slate-300 mb-2">{incident.description}</p>
                      <div className="flex items-center justify-between text-xs text-slate-400">
                        <span>📍 {incident.latitude.toFixed(4)}, {incident.longitude.toFixed(4)}</span>
                        <span>Confidence: {(incident.confidence * 100).toFixed(0)}%</span>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>

            {/* Stats Sidebar */}
            <div className="space-y-6">
              <div className="bg-slate-900 rounded-lg border border-slate-800 p-6">
                <h3 className="text-lg font-bold mb-4">Crisis Distribution</h3>
                <div className="space-y-3">
                  {crisisTypes.map((type) => {
                    const count = incidents.filter((i) => i.crisis_type === type).length
                    const percentage = (count / incidents.length) * 100

                    return (
                      <div key={type} className="space-y-2">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-2">
                            <span className="text-xl">{crisisEmojis[type] || "⚠️"}</span>
                            <span className="text-sm capitalize">{type.replace("_", " ")}</span>
                          </div>
                          <span className="text-sm font-semibold">{count}</span>
                        </div>
                        <div className="w-full bg-slate-800 rounded-full h-2">
                          <div
                            className={`h-2 rounded-full ${crisisColors[type]}`}
                            style={{ width: `${percentage}%` }}
                          />
                        </div>
                      </div>
                    )
                  })}
                </div>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Report Form Modal */}
      {showReportForm && (
        <ReportForm
          onClose={() => setShowReportForm(false)}
          onSubmit={async () => {
            await fetchIncidents()
            await fetchStats()
          }}
        />
      )}
    </div>
  )
}