"use client"

import { useEffect, useState } from "react"

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

interface IncidentMapProps {
  incidents: Incident[]
}

const crisisColors: Record<string, string> = {
  flood: '#3B82F6',
  fire: '#EF4444',
  earthquake: '#8B5CF6',
  gas_leak: '#F59E0B',
  landslide: '#78350F',
  power_outage: '#6B7280',
  storm: '#10B981',
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

export function IncidentMap({ incidents }: IncidentMapProps) {
  const [isClient, setIsClient] = useState(false)
  const [mapReady, setMapReady] = useState(false)

  useEffect(() => {
    setIsClient(true)
  }, [])

  useEffect(() => {
    if (!isClient) return

    let mapInstance: any = null
    let L: any = null

    const initMap = async () => {
      try {
        console.log('[Main Map] Initializing...')
        
        // Import Leaflet
        L = (await import('leaflet')).default
        console.log('[Main Map] Leaflet loaded')

        // Fix marker icons
        delete (L.Icon.Default.prototype as any)._getIconUrl
        L.Icon.Default.mergeOptions({
          iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
          iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
          shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
        })

        const container = document.getElementById('main-crisis-map')
        if (!container) {
          console.error('[Main Map] Container not found!')
          return
        }

        console.log('[Main Map] Container found')

        // Remove existing map if any
        if ((container as any)._leaflet_id) {
          console.log('[Main Map] Clearing existing map')
          delete (container as any)._leaflet_id
        }

        // Create map
        console.log('[Main Map] Creating map instance')
        mapInstance = L.map('main-crisis-map').setView([43.6532, -79.3832], 12)

        console.log('[Main Map] Adding tiles')
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
          attribution: '&copy; OpenStreetMap',
          maxZoom: 19,
        }).addTo(mapInstance)

        console.log('[Main Map] Map created successfully!')

        // Add all incidents
        console.log(`[Main Map] Adding ${incidents.length} incidents`)
        incidents.forEach((incident, index) => {
          const color = crisisColors[incident.crisis_type] || '#6B7280'
          const emoji = crisisEmojis[incident.crisis_type] || "⚠️"

          console.log(`[Main Map] Adding incident ${index + 1}: ${incident.crisis_type}`)

          // Add circle
          L.circle([incident.latitude, incident.longitude], {
            color: color,
            fillColor: color,
            fillOpacity: 0.3,
            radius: 200,
            weight: 2,
          }).addTo(mapInstance)

          // Add marker
          const marker = L.marker([incident.latitude, incident.longitude]).addTo(mapInstance)
          
          const popupContent = `
            <div style="padding: 12px; min-width: 250px;">
              <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                <span style="font-size: 28px;">${emoji}</span>
                <h3 style="font-weight: bold; margin: 0; text-transform: capitalize;">
                  ${incident.crisis_type.replace('_', ' ')}
                </h3>
              </div>
              <p style="margin: 8px 0; font-size: 14px;">${incident.description}</p>
              <div style="font-size: 12px; color: #666; margin-top: 8px;">
                <div style="margin: 4px 0;"><strong>Status:</strong> 
                  <span style="color: ${incident.status === 'active' ? '#ef4444' : '#22c55e'}; font-weight: bold;">
                    ${incident.status.toUpperCase()}
                  </span>
                </div>
                <div style="margin: 4px 0;"><strong>Confidence:</strong> ${(incident.confidence * 100).toFixed(1)}%</div>
                <div style="margin: 4px 0;"><strong>Reported:</strong> ${new Date(incident.timestamp).toLocaleString()}</div>
                <div style="margin: 4px 0;"><strong>Source:</strong> ${incident.source}</div>
              </div>
            </div>
          `
          
          marker.bindPopup(popupContent)
        })

        // Force map to resize
        setTimeout(() => {
          if (mapInstance) {
            mapInstance.invalidateSize()
            console.log('[Main Map] Map size refreshed')
          }
        }, 100)

        setMapReady(true)
        console.log('[Main Map] Initialization complete!')

      } catch (error) {
        console.error('[Main Map] Error:', error)
      }
    }

    const timer = setTimeout(initMap, 100)

    return () => {
      clearTimeout(timer)
      if (mapInstance) {
        try {
          mapInstance.remove()
          console.log('[Main Map] Cleaned up')
        } catch (e) {
          console.log('[Main Map] Cleanup error (safe to ignore)')
        }
      }
    }
  }, [isClient, incidents]) // Re-initialize when incidents change

  if (!isClient) {
    return (
      <div className="w-full h-[600px] bg-slate-800 rounded-lg flex items-center justify-center">
        <div className="text-center">
          <div className="text-4xl mb-3">🗺️</div>
          <p className="text-slate-400">Initializing map...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="relative w-full h-[600px]">
      {/* Leaflet CSS */}
      <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/leaflet.css" />
      
      {/* Map Container */}
      <div 
        id="main-crisis-map" 
        className="w-full h-full rounded-lg bg-slate-800"
        style={{ zIndex: 0 }}
      />

      {/* No incidents overlay */}
      {incidents.length === 0 && (
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none z-10 bg-slate-900/50 backdrop-blur-sm rounded-lg">
          <div className="text-center p-8 bg-slate-800 rounded-lg border border-slate-700">
            <div className="text-5xl mb-4">🗺️</div>
            <h3 className="text-xl font-bold mb-2">No Incidents Reported</h3>
            <p className="text-sm text-slate-400">
              Click "+ Report Incident" to submit a crisis report
            </p>
          </div>
        </div>
      )}

      {/* Incident counter */}
      {incidents.length > 0 && mapReady && (
        <div className="absolute top-4 left-4 bg-slate-900/90 backdrop-blur border border-slate-700 rounded-lg px-4 py-3 z-[1000] shadow-lg">
          <div className="flex items-center gap-3">
            <span className="text-3xl">📍</span>
            <div>
              <div className="text-xs text-slate-400 uppercase tracking-wide">Active Incidents</div>
              <div className="text-2xl font-bold text-white">{incidents.length}</div>
            </div>
          </div>
        </div>
      )}

      {/* Legend */}
      <div className="absolute top-4 right-4 bg-slate-900/90 backdrop-blur border border-slate-700 rounded-lg p-4 z-[1000] shadow-lg">
        <h4 className="text-sm font-bold mb-3 text-white">Crisis Types</h4>
        <div className="space-y-2">
          {Object.entries(crisisColors).map(([type, color]) => (
            <div key={type} className="flex items-center gap-2 text-xs">
              <div 
                className="w-4 h-4 rounded-full border-2 border-white/20" 
                style={{ backgroundColor: color }}
              />
              <span className="text-slate-300 capitalize">{type.replace('_', ' ')}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Loading indicator */}
      {!mapReady && incidents.length > 0 && (
        <div className="absolute inset-0 flex items-center justify-center z-10 bg-slate-900/50 backdrop-blur-sm rounded-lg">
          <div className="text-center p-6 bg-slate-800 rounded-lg border border-slate-700">
            <div className="animate-spin text-4xl mb-3">🔄</div>
            <p className="text-slate-300">Loading map...</p>
          </div>
        </div>
      )}
    </div>
  )
}