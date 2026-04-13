# AP RF Metrics & Charts Implementation

## Summary
Add RF throughput and client/channel metrics for Access Points with synchronized charts using the same zoom/pan/scroll behavior as WAN metrics. Uses tabbed radio band selection (2.4GHz, 5GHz, 6GHz, All Bands).

## Requirements

### RF Throughput Chart (Top)
- **Tabbed view**: `[2.4 GHz] [5 GHz] [6 GHz] [All Bands]`
- **Default**: "All Bands" (combined RX/TX across enabled radios)
- **Per-band view**: RX/TX lines (same as WAN but no error counters)
- **Hide tabs for unavailable bands**: x20 series has no 6GHz
- **Same chart behaviors**: zoom, pan, scroll, time range selector, summary cards

### Client & Channel Utilization Chart (Bottom, Synchronized)
- **Left axis**: Client count (per selected band or total)
- **Right axis**: Channel utilization % (current operating channel only)
- **Synced**: Same zoom/pan state as throughput chart above

### Backend Infrastructure
- **Polling interval**: 2 seconds (matches WAN, per `POLL_INTERVAL_RF` config)
- **Database**: Same pattern as `metrics_wan` (hypertable + continuous aggregates)
- **Retention**: Same policies (raw 90d, aggregates 90d-2yr)
- **API**: Extend `/api/v1/metrics` with `metric_type=rf`

---

## API Data Available (from x30 API spec)

`GET /v2/wireless/radiostatus` returns per radio:

| Field | Type | Description |
|-------|------|-------------|
| `interface` | enum | `2.4GHZ`, `5GHZ`, `6GHZ` |
| `enabled` | boolean | Radio on/off |
| `rx` | integer | Cumulative RX bytes |
| `tx` | integer | Cumulative TX bytes |
| `clients` | integer | Connected clients (0-200) |
| `guestClients` | integer | Guest clients (v2 only) |

`GET /v1/wireless/channelUtil?interface=2.4GHZ|5GHZ|6GHZ` returns:
```json
{
  "channelUtil": "45",  // String percentage
  "deviceId": "D4:6A:91:12:34:56",
  "errCode": 0,
  "message": "OK"
}
```
Note: Must call once per band (3 API calls).

**No error counters available** - confirmed from API spec.

---

## Phase 1: Database Schema

### File: `database/migrations/2025XXXX_create_metrics_rf.sql`

Create `metrics_rf` hypertable following `metrics_wan` pattern:

```sql
CREATE TABLE metrics_rf (
    time TIMESTAMPTZ NOT NULL,
    device_id UUID NOT NULL,
    radio_band VARCHAR(20) NOT NULL,        -- '2.4GHz', '5GHz', '6GHz'

    -- Traffic counters (cumulative)
    rx_bytes BIGINT NOT NULL DEFAULT 0,
    tx_bytes BIGINT NOT NULL DEFAULT 0,

    -- Client metrics
    client_count INTEGER NOT NULL DEFAULT 0,
    guest_client_count INTEGER NOT NULL DEFAULT 0,

    -- Channel metrics
    channel INTEGER,                         -- Operating channel number
    channel_util_pct DOUBLE PRECISION,       -- Channel utilization %

    -- Radio state
    enabled BOOLEAN NOT NULL DEFAULT true,

    CONSTRAINT chk_mrf_band CHECK (radio_band IN ('2.4GHz', '5GHz', '6GHz'))
);

-- Convert to hypertable
SELECT create_hypertable('metrics_rf', 'time', chunk_time_interval => INTERVAL '1 day');

-- Indexes
CREATE UNIQUE INDEX uq_metrics_rf_point ON metrics_rf (device_id, radio_band, time);
CREATE INDEX idx_metrics_rf_device_band_time ON metrics_rf (device_id, radio_band, time DESC);

-- Enable compression
ALTER TABLE metrics_rf SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'device_id, radio_band',
    timescaledb.compress_orderby = 'time DESC'
);
```

### File: `database/migrations/2025XXXX_create_metrics_rf_aggregates.sql`

Create continuous aggregates (1min, 5min, 30min, 1hour):

```sql
-- 1-minute aggregate
CREATE MATERIALIZED VIEW metrics_rf_1min WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 minute', time) AS bucket,
    device_id,
    radio_band,
    (MAX(rx_bytes) - MIN(rx_bytes)) AS rx_bytes_delta,
    (MAX(tx_bytes) - MIN(tx_bytes)) AS tx_bytes_delta,
    AVG(client_count) AS client_count_avg,
    MAX(client_count) AS client_count_max,
    AVG(channel_util_pct) AS channel_util_avg,
    MAX(channel_util_pct) AS channel_util_max,
    COUNT(*) AS sample_count
FROM metrics_rf
GROUP BY bucket, device_id, radio_band;

-- Similar for 5min, 30min, 1hour...
```

### File: `database/migrations/2025XXXX_metrics_rf_policies.sql`

Same retention policies as WAN:
- Raw: 90 days
- 1min/5min/30min: 90 days
- 1hour: 180 days
- Compression: after 7 days

---

## Phase 2: Backend Implementation

### File: `backend/internal/devices/ap/client.go`

Add `GetRFMetrics()` method:

```go
type RadioStatus struct {
    Interface    string `json:"interface"`    // "2.4GHZ", "5GHZ", "6GHZ"
    Enabled      bool   `json:"enabled"`
    RxBytes      int64  `json:"rx"`
    TxBytes      int64  `json:"tx"`
    Networks     int    `json:"networks"`
    Clients      int    `json:"clients"`
    GuestClients int    `json:"guestClients"` // v2 only
}

type RFMetrics struct {
    Radio0    RadioStatus `json:"radio0"`
    Radio1    RadioStatus `json:"radio1"`
    Radio2    RadioStatus `json:"radio2"`
    Timestamp time.Time
}

func (c *Client) GetRFMetrics(ctx context.Context) (*RFMetrics, error) {
    // GET /v2/wireless/radiostatus
}

func (c *Client) GetChannelUtilization(ctx context.Context) (map[string]float64, error) {
    // GET /v1/wireless/channelUtil
}
```

### File: `backend/internal/poller/ap_rf_poller.go` (new)

Create RF metrics poller (separate from metadata poller):

```go
type APRFPoller struct {
    client       *ap.Client
    pool         *pgxpool.Pool
    deviceID     string
    ticker       *time.Ticker  // 2-second interval
    stateStore   *discovery.DeviceStateStore
    // ... similar to RouterPoller for WAN metrics
}

func (p *APRFPoller) collectMetrics(ctx context.Context) {
    rfMetrics, _ := p.client.GetRFMetrics(ctx)
    channelUtil, _ := p.client.GetChannelUtilization(ctx)

    // Insert into metrics_rf for each enabled radio
    for _, radio := range []RadioStatus{rfMetrics.Radio0, rfMetrics.Radio1, rfMetrics.Radio2} {
        if radio.Enabled {
            p.insertMetric(ctx, radio, channelUtil[radio.Interface])
        }
    }
}
```

### File: `backend/internal/poller/ap_manager.go`

Update to start both metadata poller (5min) and RF poller (2s):

```go
func (m *APPollerManager) startPollerForAP(ctx context.Context, ap *apDevice) error {
    // Existing metadata poller (5-minute interval)
    metadataPoller := NewAPPoller(...)

    // New RF metrics poller (2-second interval)
    rfPoller := NewAPRFPoller(...)

    m.pollers[ap.ID] = &apPollerPair{
        metadata: metadataPoller,
        rf:       rfPoller,
    }
}
```

### File: `backend/internal/handlers/metrics.go`

Extend to support `metric_type=rf`:

```go
func (h *MetricsHandler) GetMetrics(c echo.Context) error {
    metricType := c.QueryParam("metric_type") // "wan" or "rf"

    switch metricType {
    case "rf":
        return h.getRFMetrics(c)
    default:
        return h.getWANMetrics(c)
    }
}

func (h *MetricsHandler) getRFMetrics(c echo.Context) error {
    // Same hybrid query strategy as WAN
    // Select data source based on time range
}
```

---

## Phase 3: Frontend Implementation

### File: `frontend/src/types/api.ts`

Add RF metric type:

```typescript
export interface RFMetric {
    timestamp: string;
    device_id: string;
    metric_type: "rf";
    radio_band: "2.4GHz" | "5GHz" | "6GHz";
    rx_bytes: number;
    tx_bytes: number;
    rx_bytes_per_sec?: number;
    tx_bytes_per_sec?: number;
    client_count: number;
    guest_client_count: number;
    channel?: number;
    channel_util_pct?: number;
    enabled: boolean;
}
```

### File: `frontend/src/components/charts/RFMetricsChart.tsx` (new)

Fork of MetricsChart with:
- Band tab selector: `[2.4 GHz] [5 GHz] [6 GHz] [All Bands]`
- No error counters (only RX/TX)
- Filters/aggregates data by selected band

```typescript
interface RFMetricsChartProps {
    deviceId: string;
    timeRange: TimeRange;
    zoomState?: ZoomState;
    onZoomChange?: (zoom: ZoomState) => void;
    // ... same as MetricsChart
}

type RadioBand = "2.4GHz" | "5GHz" | "6GHz" | "all";

export function RFMetricsChart({ deviceId, ... }: RFMetricsChartProps) {
    const [selectedBand, setSelectedBand] = useState<RadioBand>("all");
    const [availableBands, setAvailableBands] = useState<RadioBand[]>([]);

    // Filter metrics by selected band
    // Aggregate if "all" selected
    // Hide tabs for unavailable bands (x20 series = no 6GHz)
}
```

### File: `frontend/src/components/charts/RFQualityChart.tsx` (new)

Client count + channel utilization chart:

```typescript
export function RFQualityChart({ deviceId, timeRange, zoomState, ... }) {
    // Left axis: Client count
    // Right axis: Channel utilization %
    // Same sync pattern as NetworkQualityChart
}
```

### File: `frontend/src/components/charts/RFMetricsContainer.tsx` (new)

Container managing synchronized charts:

```typescript
export function RFMetricsContainer({ deviceId }: { deviceId: string }) {
    const [timeRange, setTimeRange] = useState<TimeRange>("5m");
    const [zoomState, setZoomState] = useState<ZoomState>({ start: 0, end: 100 });

    return (
        <div className="rounded-lg border border-[var(--border)] bg-[var(--card-bg)]">
            {/* Header with time range selector */}
            <div className="flex items-center justify-between border-b p-4">
                <h2>RF Metrics</h2>
                <TimeRangeSelector value={timeRange} onChange={setTimeRange} />
            </div>

            {/* RF Throughput Chart (with band tabs) */}
            <RFMetricsChart
                deviceId={deviceId}
                timeRange={timeRange}
                zoomState={zoomState}
                onZoomChange={setZoomState}
                showXAxisLabels={false}
            />

            {/* Client & Channel Utilization Chart */}
            <RFQualityChart
                deviceId={deviceId}
                timeRange={timeRange}
                zoomState={zoomState}
                onZoomChange={setZoomState}
            />
        </div>
    );
}
```

### File: `frontend/src/app/devices/[id]/page.tsx`

Add RFMetricsContainer to AP overview page:

```typescript
export default function DeviceOverviewPage() {
    // ... existing code

    return (
        <div className="space-y-6">
            {/* Existing device info cards */}

            {/* Add RF metrics for access points */}
            {device.device_type === "access_point" && (
                <RFMetricsContainer deviceId={deviceId} />
            )}
        </div>
    );
}
```

---

## Files Summary

| File | Action | Purpose |
|------|--------|---------|
| `database/migrations/2025XXXX_create_metrics_rf.sql` | Create | RF hypertable schema |
| `database/migrations/2025XXXX_create_metrics_rf_aggregates.sql` | Create | Continuous aggregates |
| `database/migrations/2025XXXX_metrics_rf_policies.sql` | Create | Retention/compression policies |
| `backend/internal/devices/ap/client.go` | Edit | Add GetRFMetrics(), GetChannelUtilization() |
| `backend/internal/poller/ap_rf_poller.go` | Create | RF metrics poller (2s interval) |
| `backend/internal/poller/ap_manager.go` | Edit | Start RF poller alongside metadata poller |
| `backend/internal/handlers/metrics.go` | Edit | Add metric_type=rf support |
| `frontend/src/types/api.ts` | Edit | Add RFMetric type |
| `frontend/src/components/charts/RFMetricsChart.tsx` | Create | Tabbed RF throughput chart |
| `frontend/src/components/charts/RFQualityChart.tsx` | Create | Client count + channel util chart |
| `frontend/src/components/charts/RFMetricsContainer.tsx` | Create | Synchronized chart container |
| `frontend/src/app/devices/[id]/page.tsx` | Edit | Add RF charts to AP overview |
| `backend/internal/handlers/sse.go` | Edit | Add BroadcastRFMetrics() |
| `backend/cmd/server/main.go` | Edit | Add RF to metrics broadcaster |

---

## Implementation Order

1. **Database migrations** - Create metrics_rf schema + aggregates + policies
2. **AP client methods** - GetRFMetrics, GetChannelUtilization
3. **RF poller** - 2-second polling, insert to metrics_rf
4. **AP manager update** - Start RF poller with APs
5. **Metrics handler** - Add RF query support
6. **SSE broadcaster** - Add RF to real-time streaming
7. **Frontend types** - RFMetric interface
8. **RFMetricsChart** - Tabbed throughput chart
9. **RFQualityChart** - Client/channel chart
10. **RFMetricsContainer** - Synchronized container
11. **AP page integration** - Add charts to AP overview

---

## Decisions Made

1. **SSE Streaming**: ✅ Yes, add RF to SSE broadcast for real-time 5m view (matches WAN)
2. **Channel Utilization Polling**: ✅ Every 2 seconds alongside RF metrics (consistent data)

---

## Additional Backend Work (SSE)

### File: `backend/internal/handlers/sse.go`

Add RF metrics to broadcast loop:

```go
func (h *SSEHandler) BroadcastRFMetrics(metrics []RFMetric) {
    // Same pattern as BroadcastMetrics for WAN
    // event: rf_metrics
}
```

### File: `backend/cmd/server/main.go`

Extend `startMetricsBroadcaster` to include RF:

```go
// Query latest RF metrics (same pattern as WAN)
// Broadcast via sseHandler.BroadcastRFMetrics()
```
