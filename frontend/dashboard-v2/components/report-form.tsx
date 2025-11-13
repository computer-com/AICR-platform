"use client"

import React, { useState, useEffect } from "react"

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"

interface ReportFormProps {
  onClose: () => void
  onSubmit: () => void
}

export function ReportForm({ onClose, onSubmit }: ReportFormProps) {
  const [formData, setFormData] = useState({
    description: "",
    latitude: "43.6532",
    longitude: "-79.3832",
  })
  const [loading, setLoading] = useState(false)
  const [map, setMap] = useState<any>(null)
  const [marker, setMarker] = useState<any>(null)
  const [searchAddress, setSearchAddress] = useState("")
  const [searching, setSearching] = useState(false)

  // Initialize mini map for location selection
  useEffect(() => {
    let mapInstance: any = null
    let markerInstance: any = null

    const initMap = async () => {
      const L = (await import('leaflet')).default

      // Fix marker icons
      delete (L.Icon.Default.prototype as any)._getIconUrl
      L.Icon.Default.mergeOptions({
        iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
        iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
        shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
      })

      const container = document.getElementById('location-picker-map')
      if (!container) return

      // Remove existing map if any
      if ((container as any)._leaflet_id) {
        delete (container as any)._leaflet_id
      }

      mapInstance = L.map('location-picker-map').setView(
        [parseFloat(formData.latitude), parseFloat(formData.longitude)], 
        13
      )

      L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; OpenStreetMap'
      }).addTo(mapInstance)

      // Add draggable marker
      markerInstance = L.marker(
        [parseFloat(formData.latitude), parseFloat(formData.longitude)],
        { draggable: true }
      ).addTo(mapInstance)

      // Update coordinates when marker is dragged
      markerInstance.on('dragend', function(e: any) {
        const position = e.target.getLatLng()
        setFormData(prev => ({
          ...prev,
          latitude: position.lat.toFixed(4),
          longitude: position.lng.toFixed(4)
        }))
      })

      // Click on map to move marker
      mapInstance.on('click', function(e: any) {
        const { lat, lng } = e.latlng
        markerInstance.setLatLng([lat, lng])
        setFormData(prev => ({
          ...prev,
          latitude: lat.toFixed(4),
          longitude: lng.toFixed(4)
        }))
      })

      setMap(mapInstance)
      setMarker(markerInstance)
    }

    const timer = setTimeout(initMap, 100)

    return () => {
      clearTimeout(timer)
      if (mapInstance) {
        mapInstance.remove()
      }
    }
  }, [])

  // Update marker when coordinates change manually
  useEffect(() => {
    if (marker && map) {
      const lat = parseFloat(formData.latitude)
      const lng = parseFloat(formData.longitude)
      if (!isNaN(lat) && !isNaN(lng)) {
        marker.setLatLng([lat, lng])
        map.setView([lat, lng], map.getZoom())
      }
    }
  }, [formData.latitude, formData.longitude, marker, map])

  // Search address using Nominatim (OpenStreetMap)
  const handleSearchAddress = async () => {
    if (!searchAddress.trim()) return

    setSearching(true)
    try {
      const response = await fetch(
        `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(searchAddress)}&limit=1`
      )
      const data = await response.json()

      if (data && data.length > 0) {
        const { lat, lon } = data[0]
        setFormData(prev => ({
          ...prev,
          latitude: parseFloat(lat).toFixed(4),
          longitude: parseFloat(lon).toFixed(4)
        }))
        
        // Move map and marker
        if (map && marker) {
          map.setView([lat, lon], 15)
          marker.setLatLng([lat, lon])
        }

        alert(`Location found: ${data[0].display_name}`)
      } else {
        alert('Address not found. Please try a different search.')
      }
    } catch (error) {
      console.error('Error searching address:', error)
      alert('Error searching address. Please try again.')
    } finally {
      setSearching(false)
    }
  }

  // Get current location
  const handleGetCurrentLocation = () => {
    if ('geolocation' in navigator) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          const lat = position.coords.latitude.toFixed(4)
          const lng = position.coords.longitude.toFixed(4)
          setFormData(prev => ({
            ...prev,
            latitude: lat,
            longitude: lng
          }))

          if (map && marker) {
            map.setView([lat, lng], 15)
            marker.setLatLng([lat, lng])
          }

          alert('Current location set!')
        },
        (error) => {
          alert('Unable to get current location. Please enable location services.')
          console.error('Geolocation error:', error)
        }
      )
    } else {
      alert('Geolocation is not supported by your browser.')
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)

    try {
      const response = await fetch(`${API_URL}/report`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          description: formData.description,
          latitude: parseFloat(formData.latitude),
          longitude: parseFloat(formData.longitude),
        }),
      })
      const data = await response.json()
      alert(`Report submitted! Predicted crisis type: ${data.crisis_type}`)
      onSubmit()
      onClose()
    } catch (error) {
      console.error("Error submitting report:", error)
      alert("Error submitting report. Please try again.")
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4 overflow-y-auto">
      <div className="w-full max-w-4xl my-8 bg-slate-900 rounded-lg border border-slate-700 shadow-2xl">
        <div className="p-6 border-b border-slate-700">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-xl font-bold">Report New Incident</h2>
              <p className="text-sm text-slate-400">Submit a crisis report for AI analysis</p>
            </div>
            <button
              onClick={onClose}
              className="text-slate-400 hover:text-white transition text-2xl"
            >
              ✕
            </button>
          </div>
        </div>
        
        <div className="p-6">
          <form onSubmit={handleSubmit} className="space-y-6">
            {/* Description */}
            <div>
              <label htmlFor="description" className="block text-sm font-medium mb-2">
                Crisis Description *
              </label>
              <textarea
                id="description"
                name="description"
                required
                minLength={10}
                value={formData.description}
                onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg focus:border-blue-500 focus:outline-none"
                rows={4}
                placeholder="Describe the crisis situation in detail..."
                aria-label="Crisis description"
              />
            </div>

            {/* Location Section */}
            <div className="space-y-4">
              <h3 className="text-lg font-semibold">Select Location</h3>
              
              {/* Address Search */}
              <div>
                <label className="block text-sm font-medium mb-2">Search by Address</label>
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={searchAddress}
                    onChange={(e) => setSearchAddress(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && (e.preventDefault(), handleSearchAddress())}
                    placeholder="Enter street address, city, or landmark..."
                    className="flex-1 px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg focus:border-blue-500 focus:outline-none"
                  />
                  <button
                    type="button"
                    onClick={handleSearchAddress}
                    disabled={searching}
                    className="px-6 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg font-medium transition disabled:opacity-50"
                  >
                    {searching ? '🔍 Searching...' : '🔍 Search'}
                  </button>
                  <button
                    type="button"
                    onClick={handleGetCurrentLocation}
                    className="px-6 py-2 bg-green-600 hover:bg-green-700 rounded-lg font-medium transition"
                    title="Use my current location"
                  >
                    📍 My Location
                  </button>
                </div>
              </div>

              {/* Interactive Map */}
              <div>
                <label className="block text-sm font-medium mb-2">
                  Click on map or drag marker to select location
                </label>
                <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/leaflet.css" />
                <div 
                  id="location-picker-map" 
                  className="w-full h-[300px] rounded-lg border border-slate-700"
                  style={{ zIndex: 1 }}
                />
                <p className="text-xs text-slate-400 mt-2">
                  💡 Tip: You can click anywhere on the map or drag the red marker
                </p>
              </div>

              {/* Manual Coordinates */}
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label htmlFor="latitude" className="block text-sm font-medium mb-2">
                    Latitude *
                  </label>
                  <input
                    id="latitude"
                    name="latitude"
                    required
                    type="number"
                    step="0.0001"
                    value={formData.latitude}
                    onChange={(e) => setFormData({ ...formData, latitude: e.target.value })}
                    className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg focus:border-blue-500 focus:outline-none"
                    aria-label="Latitude coordinate"
                  />
                </div>
                <div>
                  <label htmlFor="longitude" className="block text-sm font-medium mb-2">
                    Longitude *
                  </label>
                  <input
                    id="longitude"
                    name="longitude"
                    required
                    type="number"
                    step="0.0001"
                    value={formData.longitude}
                    onChange={(e) => setFormData({ ...formData, longitude: e.target.value })}
                    className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg focus:border-blue-500 focus:outline-none"
                    aria-label="Longitude coordinate"
                  />
                </div>
              </div>
            </div>
            
            {/* Submit Buttons */}
            <div className="flex gap-3 pt-4">
              <button
                type="button"
                onClick={onClose}
                className="flex-1 px-6 py-3 bg-slate-800 hover:bg-slate-700 rounded-lg font-medium transition"
              >
                Cancel
              </button>
              <button
                type="submit"
                disabled={loading}
                className="flex-1 px-6 py-3 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg font-medium hover:opacity-90 transition disabled:opacity-50"
              >
                {loading ? "Submitting..." : "🚨 Submit Report"}
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  )
}