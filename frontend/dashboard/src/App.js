import React, { useState, useEffect } from 'react';
import { MapContainer, TileLayer, Marker, Popup, Circle } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';

// Fix for default marker icons in React Leaflet
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: require('leaflet/dist/images/marker-icon-2x.png'),
  iconUrl: require('leaflet/dist/images/marker-icon.png'),
  shadowUrl: require('leaflet/dist/images/marker-shadow.png'),
});

// Custom marker colors for different crisis types
const crisisColors = {
  flood: '#3B82F6',      // Blue
  fire: '#EF4444',       // Red
  earthquake: '#8B5CF6', // Purple
  gas_leak: '#F59E0B',   // Orange
  landslide: '#78350F',  // Brown
  power_outage: '#6B7280', // Gray
  storm: '#10B981'       // Green
};

const API_URL = 'http://localhost:8000';

function App() {
  const [incidents, setIncidents] = useState([]);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [selectedFilter, setSelectedFilter] = useState('all');
  const [showReportForm, setShowReportForm] = useState(false);
  const [reportForm, setReportForm] = useState({
    description: '',
    latitude: 43.6532,
    longitude: -79.3832
  });

  // Fetch incidents from API
  const fetchIncidents = async () => {
    try {
      const response = await fetch(`${API_URL}/incidents`);
      const data = await response.json();
      setIncidents(data);
    } catch (error) {
      console.error('Error fetching incidents:', error);
    }
  };

  // Fetch statistics
  const fetchStats = async () => {
    try {
      const response = await fetch(`${API_URL}/stats`);
      const data = await response.json();
      setStats(data);
    } catch (error) {
      console.error('Error fetching stats:', error);
    }
  };

  // Initial data load
  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      await Promise.all([fetchIncidents(), fetchStats()]);
      setLoading(false);
    };
    loadData();

    // Auto-refresh every 10 seconds
    const interval = setInterval(() => {
      fetchIncidents();
      fetchStats();
    }, 10000);

    return () => clearInterval(interval);
  }, []);

  // Submit new crisis report
  const handleSubmitReport = async (e) => {
    e.preventDefault();
    try {
      const response = await fetch(`${API_URL}/report`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(reportForm)
      });
      const data = await response.json();
      alert(`Report submitted! Predicted crisis type: ${data.crisis_type}`);
      setReportForm({ description: '', latitude: 43.6532, longitude: -79.3832 });
      setShowReportForm(false);
      fetchIncidents();
      fetchStats();
    } catch (error) {
      alert('Error submitting report: ' + error.message);
    }
  };

  // Filter incidents
  const filteredIncidents = selectedFilter === 'all'
    ? incidents
    : incidents.filter(inc => inc.crisis_type === selectedFilter);

  // Get unique crisis types
  const crisisTypes = [...new Set(incidents.map(inc => inc.crisis_type))];

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen bg-gray-100">
        <div className="text-xl font-semibold">Loading AICR Platform...</div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-blue-600 text-white shadow-lg z-10">
        <div className="px-6 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold">🚨 AICR Crisis Response Platform</h1>
              <p className="text-sm text-blue-100">AI-Augmented Local Crisis Management</p>
            </div>
            <button
              onClick={() => setShowReportForm(!showReportForm)}
              className="bg-white text-blue-600 px-6 py-2 rounded-lg font-semibold hover:bg-blue-50 transition"
            >
              {showReportForm ? 'Close Form' : '+ New Report'}
            </button>
          </div>
        </div>

        {/* Statistics Bar */}
        {stats && (
          <div className="bg-blue-700 px-6 py-3 flex gap-8 text-sm">
            <div>
              <span className="text-blue-200">Total Incidents:</span>
              <span className="ml-2 font-bold">{stats.total_incidents}</span>
            </div>
            <div>
              <span className="text-blue-200">Active:</span>
              <span className="ml-2 font-bold">{stats.active_incidents}</span>
            </div>
            <div>
              <span className="text-blue-200">Resolved:</span>
              <span className="ml-2 font-bold">{stats.resolved_incidents}</span>
            </div>
            <div>
              <span className="text-blue-200">Avg Confidence:</span>
              <span className="ml-2 font-bold">{(stats.avg_confidence * 100).toFixed(1)}%</span>
            </div>
          </div>
        )}
      </header>

      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar */}
        <aside className="w-80 bg-white shadow-lg overflow-y-auto">
          {/* Report Form */}
          {showReportForm && (
            <div className="p-4 border-b bg-blue-50">
              <h3 className="font-bold text-lg mb-3">Submit Crisis Report</h3>
              <div className="space-y-3">
                <div>
                  <label className="block text-sm font-medium mb-1">Description</label>
                  <textarea
                    required
                    value={reportForm.description}
                    onChange={(e) => setReportForm({ ...reportForm, description: e.target.value })}
                    className="w-full border rounded px-3 py-2 text-sm"
                    rows="3"
                    placeholder="Describe the crisis situation..."
                  />
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <label className="block text-sm font-medium mb-1">Latitude</label>
                    <input
                      required
                      type="number"
                      step="0.0001"
                      value={reportForm.latitude}
                      onChange={(e) => setReportForm({ ...reportForm, latitude: parseFloat(e.target.value) })}
                      className="w-full border rounded px-3 py-2 text-sm"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium mb-1">Longitude</label>
                    <input
                      required
                      type="number"
                      step="0.0001"
                      value={reportForm.longitude}
                      onChange={(e) => setReportForm({ ...reportForm, longitude: parseFloat(e.target.value) })}
                      className="w-full border rounded px-3 py-2 text-sm"
                    />
                  </div>
                </div>
                <button
                  onClick={handleSubmitReport}
                  className="w-full bg-blue-600 text-white py-2 rounded font-semibold hover:bg-blue-700 transition"
                >
                  Submit Report
                </button>
              </div>
            </div>
          )}

          {/* Filters */}
          <div className="p-4 border-b">
            <h3 className="font-bold text-sm mb-2">Filter by Crisis Type</h3>
            <div className="space-y-1">
              <button
                onClick={() => setSelectedFilter('all')}
                className={`w-full text-left px-3 py-2 rounded text-sm ${
                  selectedFilter === 'all' ? 'bg-blue-100 text-blue-700 font-medium' : 'hover:bg-gray-100'
                }`}
              >
                All Incidents ({incidents.length})
              </button>
              {crisisTypes.map(type => (
                <button
                  key={type}
                  onClick={() => setSelectedFilter(type)}
                  className={`w-full text-left px-3 py-2 rounded text-sm flex items-center ${
                    selectedFilter === type ? 'bg-blue-100 text-blue-700 font-medium' : 'hover:bg-gray-100'
                  }`}
                >
                  <span
                    className="w-3 h-3 rounded-full mr-2"
                    style={{ backgroundColor: crisisColors[type] || '#gray' }}
                  />
                  {type.replace('_', ' ')} ({incidents.filter(i => i.crisis_type === type).length})
                </button>
              ))}
            </div>
          </div>

          {/* Incidents List */}
          <div className="p-4">
            <h3 className="font-bold text-sm mb-3">Recent Incidents</h3>
            <div className="space-y-3">
              {filteredIncidents.length === 0 ? (
                <p className="text-sm text-gray-500">No incidents found</p>
              ) : (
                filteredIncidents.slice(0, 20).map(incident => (
                  <div key={incident.id} className="border rounded p-3 hover:bg-gray-50 cursor-pointer">
                    <div className="flex items-start justify-between mb-1">
                      <span
                        className="px-2 py-1 rounded text-xs font-semibold text-white"
                        style={{ backgroundColor: crisisColors[incident.crisis_type] || '#gray' }}
                      >
                        {incident.crisis_type.replace('_', ' ').toUpperCase()}
                      </span>
                      <span className="text-xs text-gray-500">
                        {new Date(incident.timestamp).toLocaleTimeString()}
                      </span>
                    </div>
                    <p className="text-sm mb-2">{incident.description.substring(0, 100)}...</p>
                    <div className="flex items-center justify-between text-xs text-gray-500">
                      <span>Confidence: {(incident.confidence * 100).toFixed(0)}%</span>
                      <span className={`px-2 py-1 rounded ${
                        incident.status === 'active' ? 'bg-yellow-100 text-yellow-700' : 'bg-green-100 text-green-700'
                      }`}>
                        {incident.status}
                      </span>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        </aside>

        {/* Map */}
        <main className="flex-1">
          <MapContainer
            center={[43.6532, -79.3832]}
            zoom={12}
            style={{ height: '100%', width: '100%' }}
          >
            <TileLayer
              attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            />
            
            {filteredIncidents.map(incident => (
              <React.Fragment key={incident.id}>
                <Circle
                  center={[incident.latitude, incident.longitude]}
                  radius={200}
                  pathOptions={{
                    fillColor: crisisColors[incident.crisis_type] || '#gray',
                    fillOpacity: 0.2,
                    color: crisisColors[incident.crisis_type] || '#gray',
                    weight: 2
                  }}
                />
                <Marker position={[incident.latitude, incident.longitude]}>
                  <Popup>
                    <div className="p-2">
                      <h3 className="font-bold text-lg mb-2 capitalize">
                        {incident.crisis_type.replace('_', ' ')}
                      </h3>
                      <p className="text-sm mb-2">{incident.description}</p>
                      <div className="text-xs text-gray-600 space-y-1">
                        <div>Confidence: {(incident.confidence * 100).toFixed(1)}%</div>
                        <div>Status: {incident.status}</div>
                        <div>Time: {new Date(incident.timestamp).toLocaleString()}</div>
                        <div>Source: {incident.source}</div>
                      </div>
                    </div>
                  </Popup>
                </Marker>
              </React.Fragment>
            ))}
          </MapContainer>
        </main>
      </div>
    </div>
  );
}

export default App;