# Data Retention Settings Plan

## Overview

Add configurable data retention settings to the Settings page, allowing users to independently control how long raw data and aggregate data are stored in TimescaleDB.

---

## Current State

### Database Retention (from migration 20260106000069)

All tables are currently set to **32 days** retention:

**Raw Hypertables (8 tables):**
- `metrics_wan` - WAN interface metrics (2s polling)
- `metrics_iface` - Router interface metrics
- `metrics_rf` - AP RF metrics (2s polling)
- `metrics_client_traffic` - Per-client bandwidth
- `metrics_switch_port` - Switch port counters
- `metrics_power` - WattBox power metrics
- `metrics_outlet_power` - Per-outlet power metrics
- `metrics_ups` - UPS battery/power metrics

**Continuous Aggregates (30+ views):**
- 1-minute aggregates (multiple tables)
- 5-minute aggregates (multiple tables)
- 30-minute aggregates (multiple tables)
- 1-hour aggregates (multiple tables)
- 1-day aggregates (multiple tables)
- Plus RF, power, and outlet aggregates

### Chart Time Range Mapping

| Time Range | Data Source | Resolution |
|------------|-------------|------------|
| 5M | Raw data | 2-second |
| 30M | Raw data (on-the-fly 6s bucketing) | 6-second |
| 6H | 1-minute aggregates | 1-minute |
| 12H | 5-minute aggregates | 5-minute |
| 1D | 5-minute aggregates | 5-minute |
| 1W | 30-minute aggregates | 30-minute |
| 1M | 1-hour aggregates | 1-hour |

**Important:** Both 5M and 30M views use raw data. Aggregates are only used for 6H+ views.

---

## User Requirements

1. **Raw Data Retention**: 7-180 days (affects 5M and 30M views)
2. **Aggregate Data Retention**: 31-360 days (affects 6H, 12H, 1D, 1W, 1M views)
3. **UI**: Slider with numeric input box for each setting
4. **Apply to all charts consistently**

---

## Implementation Plan

### Phase 1: Backend - Settings API

#### New Endpoints

```
GET  /api/v1/settings/retention      - Get current retention settings
PUT  /api/v1/settings/retention      - Update retention settings
```

#### Request/Response

```json
// GET /api/v1/settings/retention
{
  "raw_retention_days": 32,
  "aggregate_retention_days": 32
}

// PUT /api/v1/settings/retention (request)
{
  "raw_retention_days": 60,
  "aggregate_retention_days": 180
}

// PUT /api/v1/settings/retention (success response)
{
  "success": true,
  "env_updated": true,
  "raw_tables_updated": 8,
  "raw_tables_total": 8,
  "aggregate_tables_updated": 40,
  "aggregate_tables_total": 40,
  "failures": []
}

// PUT /api/v1/settings/retention (partial failure response)
{
  "success": false,
  "env_updated": false,
  "raw_tables_updated": 8,
  "raw_tables_total": 8,
  "aggregate_tables_updated": 38,
  "aggregate_tables_total": 40,
  "failures": [
    "metrics_ups_1min: permission denied",
    "metrics_ups_5min: permission denied"
  ]
}
```

#### Error Codes

| Status | Condition |
|--------|-----------|
| 200 | All policies updated, .env saved |
| 400 | Validation error (e.g., aggregate < raw, out of range) |
| 409 | Transient DB state prevents update |
| 500 | DB connection error, unexpected failure |

#### Storage

Add to `.env` file:
```bash
# Data Retention (days)
DATA_RETENTION_RAW_DAYS=32
DATA_RETENTION_AGGREGATE_DAYS=32
```

#### Defaults Source

Use `.env.example` as the source of truth for system defaults. This file is:
- Git-tracked (consistent across installations)
- Already contains default values for all settings
- If defaults change, just update `.env.example`

```go
// Backend reads defaults from .env.example
func (h *SettingsHandler) GetDefaults() (*SettingsDefaults, error) {
    return h.readEnvFile(h.envExamplePath)  // .env.example
}
```

#### Restore Defaults Endpoint

```
POST /api/v1/settings/restore-defaults   - Restore all settings to system defaults
```

```json
// POST /api/v1/settings/restore-defaults (response)
{
  "success": true,
  "restored": {
    "username": "araknis",
    "password_reset": true,
    "raw_retention_days": 32,
    "aggregate_retention_days": 32
  }
}
```

This endpoint:
1. Reads default values from `.env.example`
2. Applies retention policies to DB (using default values)
3. Updates `.env` with all default values (including credentials)
4. Returns what was restored

### Phase 2: Backend - Apply Retention Policies

#### Create `backend/internal/retention/manager.go`

Manager to apply retention policy changes to TimescaleDB.

**Discovery-based approach** (don't hardcode table lists):

```sql
-- Discover raw hypertables (metrics_* pattern)
SELECT hypertable_name
FROM timescaledb_information.hypertables
WHERE hypertable_name LIKE 'metrics_%'
  AND hypertable_name NOT LIKE '%_1min'
  AND hypertable_name NOT LIKE '%_5min'
  AND hypertable_name NOT LIKE '%_30min'
  AND hypertable_name NOT LIKE '%_1hour'
  AND hypertable_name NOT LIKE '%_1day';

-- Discover continuous aggregate materialization hypertables
SELECT materialization_hypertable_name, view_name
FROM timescaledb_information.continuous_aggregates
WHERE view_name LIKE 'metrics_%';
```

**Critical: Target materialization hypertable, not view name**

Continuous aggregates are views backed by a materialization hypertable. The retention policy must be applied to the materialization hypertable (found in `timescaledb_information.continuous_aggregates`), not the view name.

```go
// Example: Get correct target for retention policy
type ContinuousAggregate struct {
    ViewName                     string
    MaterializationHypertable    string  // Use THIS for retention policy
}
```

#### Known Raw Hypertables (for reference)

- metrics_wan, metrics_iface, metrics_rf, metrics_client_traffic
- metrics_switch_port, metrics_power, metrics_outlet_power, metrics_ups

#### Known Continuous Aggregates (for reference)

- wan, iface, rf, client_traffic, switch_port, power, outlet_power, ups
- Each has _1min, _5min, _30min, _1hour, _1day variants

### Phase 3: Frontend - Settings UI

#### Add to Settings Page

New section below Device Credentials:

```
+----------------------------------------------------------+
| Data Retention                                           |
|                                                          |
| Configure how long metrics data is stored. Reducing      |
| retention will delete data older than the new limit.     |
|                                                          |
| Raw Data (5M, 30M views)                                 |
| [========--------] [  32  ] days                         |
| 7 days                                       180 days    |
|                                                          |
| Aggregate Data (6H, 12H, 1D, 1W, 1M views)               |
| [===--------------] [  32  ] days                        |
| 31 days                                      360 days    |
|                                                          |
| ! Reducing retention will permanently delete older data  |
|                                                          |
| [Save Changes]                                           |
+----------------------------------------------------------+

+----------------------------------------------------------+
| Restore System Defaults                                  |
|                                                          |
| Reset all settings to factory defaults. This will:       |
| - Reset device credentials to template values            |
| - Reset data retention to 32 days (raw and aggregate)    |
|                                                          |
| [Restore Defaults]                                       |
+----------------------------------------------------------+
```

**Restore Defaults Button:**
- Shows confirmation dialog before proceeding
- Warns that credentials will be reset
- Calls POST /api/v1/settings/restore-defaults
- Reloads all settings after success

#### Component Design

- Slider + numeric input synced bidirectionally
- Input validation (enforce min/max)
- Warning when reducing values below current
- Confirmation dialog if reducing (data loss warning)

---

## Files to Create/Modify

### New Files

| File | Purpose |
|------|---------|
| `backend/internal/retention/manager.go` | TimescaleDB retention policy manager |

### Modified Files

| File | Changes |
|------|---------|
| `backend/internal/handlers/settings.go` | Add retention + restore-defaults endpoints |
| `backend/internal/config/config.go` | Add retention config fields with defaults |
| `backend/cmd/server/main.go` | Register routes, inject DB pool + .env.example path |
| `frontend/src/app/settings/page.tsx` | Add retention section + restore defaults button |
| `frontend/src/lib/httpClient.ts` | Add retention + restore-defaults API methods |
| `.env.example` | Add retention env vars (these become the defaults source) |

---

## Key Design Decisions

### 1. All-or-Nothing for .env Persistence (Best-Effort for DB)

**Order of operations:**
1. Validate inputs (400 if invalid)
2. Apply DB retention policies (best-effort, track successes/failures)
3. Only if ALL DB updates succeed → persist to `.env`
4. If ANY DB update fails → return detailed error, do NOT update .env

**Clarification:** "All-or-nothing" applies to `.env` persistence:
- If even one DB policy update fails, `.env` is NOT updated
- This ensures the UI never shows values that don't match actual DB state
- The response always includes detailed counts so you can see what was attempted

The DB operation itself is best-effort (we try all tables and report failures) rather than transactional (since TimescaleDB policy updates aren't atomic across tables).

### 2. Discovery Over Hardcoding

Use TimescaleDB info catalogs to discover tables dynamically:
- `timescaledb_information.hypertables` for raw tables
- `timescaledb_information.continuous_aggregates` for aggregate materialization tables

Benefits:
- Automatically picks up new metrics tables added in future migrations
- No maintenance burden to keep list in sync

### 3. Validation Constraints

- Raw: minimum 7 days, maximum 180 days
- Aggregate: minimum 31 days, maximum 360 days
- Aggregate must be >= raw (400 error if violated)

### 4. Data Loss Warning

When user reduces retention:
- Show warning message in UI
- Require confirmation dialog before saving
- Log the change with old/new values

### 5. Compression Policy Unchanged

Current compression (after 1 day) remains unchanged. Only retention duration changes.

---

## Implementation Order

1. Add config fields to `config.go` with defaults
2. Create retention manager `retention/manager.go`
3. Add retention endpoints to `settings.go`
4. Register routes and inject DB pool in `main.go`
5. Update frontend settings page with slider components
6. Add httpClient methods for retention API
7. Update `.env.example` with new variables
8. Test end-to-end

---

## Verification

1. **Test GET retention** - Returns current values from .env (or defaults)
2. **Test PUT retention** - Values saved to .env and policies updated in DB
3. **Test UI sliders** - Synced with input boxes, enforce min/max limits
4. **Test reduction warning** - Confirmation dialog appears when reducing
5. **Verify TimescaleDB** - Check `timescaledb_information.jobs` for updated policy intervals
6. **Test restore defaults** - Credentials and retention reset to .env.example values
7. **Test discovery** - Add a new metrics table, verify it's picked up automatically
