"use client"

interface Incident {
  id: number
  crisis_type: string
  timestamp: string
}

interface StatsChartProps {
  incidents: Incident[]
}

export function StatsChart({ incidents }: StatsChartProps) {
  // Group incidents by hour
  const hourlyData = incidents.reduce(
    (acc, incident) => {
      const hour = new Date(incident.timestamp).getHours()
      acc[hour] = (acc[hour] || 0) + 1
      return acc
    },
    {} as Record<number, number>,
  )

  const maxCount = Math.max(...Object.values(hourlyData), 1)

  // Group by crisis type
  const crisisTypeCounts = incidents.reduce(
    (acc, incident) => {
      acc[incident.crisis_type] = (acc[incident.crisis_type] || 0) + 1
      return acc
    },
    {} as Record<string, number>
  )

  // Get peak hour
  const peakHour = Object.entries(hourlyData).reduce((a, b) => (a[1] > b[1] ? a : b), ['0', 0])

  return (
    <div className="space-y-6">
      {/* Bar Chart */}
      <div className="space-y-4">
        <h3 className="text-lg font-semibold">Incidents by Hour (Last 24 Hours)</h3>
        <div className="flex items-end justify-between gap-1 h-48 bg-slate-800 p-4 rounded-lg">
          {Array.from({ length: 24 }, (_, i) => {
            const count = hourlyData[i] || 0
            const height = maxCount > 0 ? (count / maxCount) * 100 : 0

            return (
              <div key={i} className="flex-1 flex flex-col items-center gap-2">
                <div className="w-full flex items-end justify-center h-40">
                  <div
                    className="w-full bg-gradient-to-t from-blue-500 to-purple-600 rounded-t transition-all hover:opacity-80 cursor-pointer"
                    style={{ height: `${height}%`, minHeight: count > 0 ? '4px' : '0' }}
                    title={`${i}:00 - ${count} incident${count !== 1 ? 's' : ''}`}
                  />
                </div>
                <span className="text-[10px] text-slate-400">{i}</span>
              </div>
            )
          })}
        </div>
        <p className="text-xs text-slate-400 text-center">
          Hover over bars to see exact counts
        </p>
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
          <div className="space-y-2">
            <p className="text-sm text-slate-400">Peak Hour</p>
            <p className="text-3xl font-bold text-white">
              {peakHour[0]}:00
            </p>
            <p className="text-xs text-slate-400">
              {peakHour[1]} incident{peakHour[1] !== 1 ? 's' : ''}
            </p>
          </div>
        </div>
        
        <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
          <div className="space-y-2">
            <p className="text-sm text-slate-400">Total Today</p>
            <p className="text-3xl font-bold text-white">{incidents.length}</p>
            <p className="text-xs text-slate-400">
              {Object.keys(crisisTypeCounts).length} crisis types
            </p>
          </div>
        </div>
        
        <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
          <div className="space-y-2">
            <p className="text-sm text-slate-400">Avg per Hour</p>
            <p className="text-3xl font-bold text-white">
              {(incidents.length / 24).toFixed(1)}
            </p>
            <p className="text-xs text-slate-400">
              Based on 24h period
            </p>
          </div>
        </div>
      </div>

      {/* Crisis Type Breakdown */}
      <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
        <h3 className="text-lg font-semibold mb-4">Crisis Type Breakdown</h3>
        <div className="space-y-3">
          {Object.entries(crisisTypeCounts)
            .sort((a, b) => b[1] - a[1])
            .map(([type, count]) => {
              const percentage = (count / incidents.length) * 100
              const colors: Record<string, string> = {
                flood: 'bg-blue-500',
                fire: 'bg-red-500',
                earthquake: 'bg-purple-500',
                gas_leak: 'bg-yellow-500',
                landslide: 'bg-amber-700',
                power_outage: 'bg-gray-500',
                storm: 'bg-emerald-500',
              }
              const colorClass = colors[type] || 'bg-slate-500'

              return (
                <div key={type} className="space-y-2">
                  <div className="flex items-center justify-between text-sm">
                    <span className="capitalize font-medium">{type.replace('_', ' ')}</span>
                    <span className="text-slate-400">
                      {count} ({percentage.toFixed(1)}%)
                    </span>
                  </div>
                  <div className="w-full bg-slate-700 rounded-full h-2">
                    <div
                      className={`h-2 rounded-full ${colorClass} transition-all`}
                      style={{ width: `${percentage}%` }}
                    />
                  </div>
                </div>
              )
            })}
        </div>
      </div>

      {/* Time Distribution */}
      <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
        <h3 className="text-lg font-semibold mb-4">Time Distribution</h3>
        <div className="grid grid-cols-4 gap-4">
          {[
            { label: 'Night', hours: [0, 1, 2, 3, 4, 5], icon: '🌙' },
            { label: 'Morning', hours: [6, 7, 8, 9, 10, 11], icon: '🌅' },
            { label: 'Afternoon', hours: [12, 13, 14, 15, 16, 17], icon: '☀️' },
            { label: 'Evening', hours: [18, 19, 20, 21, 22, 23], icon: '🌆' },
          ].map(({ label, hours, icon }) => {
            const count = hours.reduce((sum, h) => sum + (hourlyData[h] || 0), 0)
            return (
              <div key={label} className="bg-slate-900 rounded-lg p-4 text-center">
                <div className="text-2xl mb-2">{icon}</div>
                <div className="text-sm text-slate-400 mb-1">{label}</div>
                <div className="text-2xl font-bold">{count}</div>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}