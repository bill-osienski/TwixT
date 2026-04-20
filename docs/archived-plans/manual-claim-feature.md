# Manual Claim Feature Plan

## Overview

Add a "Manual Claim" feature that allows users to claim devices that don't appear in SSDP discovery (like AN-530 wireless bridge APs). Users provide MAC address and Serial/Service Tag, then the system locates the device via the router client table, authenticates + identifies it, and claims it.

---

## Problem Statement

**Current flow:**
1. SSDP discovery → `devices` table (unclaimed) → user clicks Claim → system connects to device → marked as claimed

**Problem:**
- Devices in wireless bridge mode (like AN-530-AP-I) don't respond to SSDP
- They never appear in `devices` table, so users can't claim them
- These devices DO appear in the router's client table (`dhcp_name_cache`) via DHCP

**Solution:**
- Manual Claim: User provides MAC + Serial → System finds IP from `dhcp_name_cache` → Authenticates + identifies device → Verifies serial → Creates and claims

---

## User Flow

1. User clicks "Manually Claim Device" button on Devices page
2. Modal appears with two input fields:
   - **MAC Address** (required): e.g., `4C:13:65:11:08:00`
   - **Serial Number / Service Tag** (required): e.g., `ST12345678`
3. User clicks "Find & Claim"
4. System:
   a. Normalizes MAC (lowercase, colon-separated)
   b. Normalizes Serial (uppercase, alphanumeric only)
   c. Looks up most recent active IP from `dhcp_name_cache` by MAC
   d. Validates IP is in expected LAN range
   e. Probes device (authenticate + fetch identity) to detect type
   f. Verifies normalized serial matches device's actual serial
   g. Creates device in `devices` table as claimed (race-safe)
5. Success → Device appears in claimed devices list

---

## Implementation Plan

### Phase 1: Backend API

#### New Endpoint

```
POST /api/v1/devices/manual-claim
```

#### Request Body

```json
{
  "mac_address": "4C:13:65:11:08:00",
  "serial_number": "ST12345678"
}
```

#### Response (Success)

```json
{
  "device_id": "uuid",
  "name": "Office-AN-530",
  "type": "access_point",
  "model": "AN-530-AP-I-N",
  "ip_address": "192.168.100.100",
  "firmware_version": "1.0.0.27"
}
```

#### Error Codes (Distinct & Actionable)

| Status | Code | Description |
|--------|------|-------------|
| 400 | `invalid_mac_format` | MAC address format invalid |
| 400 | `invalid_serial_format` | Serial number format invalid |
| 404 | `device_not_found` | MAC not found in `dhcp_name_cache` (recently) |
| 403 | `ip_not_allowed` | IP address outside allowed LAN ranges |
| 401 | `auth_failed` | Could not authenticate with device (wrong credentials) |
| 409 | `already_claimed` | Device already exists as claimed |
| 409 | `claim_in_progress` | Another claim operation is in progress for this MAC |
| 422 | `serial_mismatch` | Provided serial doesn't match device's actual serial |
| 503 | `device_unreachable` | Could not connect to device (timeout/refused) |
| 500 | `probe_failed` | Connected but failed to identify device type |

### Phase 2: Input Normalization

#### MAC Normalization

```go
// normalizeMAC converts MAC to lowercase colon-separated format
// Accepts: 4C:13:65:11:08:00, 4c-13-65-11-08-00, 4c1365110800
// Returns: 4c:13:65:11:08:00
func normalizeMAC(input string) (string, error) {
    // Remove separators and whitespace
    clean := strings.ReplaceAll(strings.ReplaceAll(strings.TrimSpace(input), ":", ""), "-", "")
    clean = strings.ReplaceAll(clean, " ", "")
    if len(clean) != 12 {
        return "", errors.New("invalid MAC address length")
    }
    // Validate hex
    if _, err := hex.DecodeString(clean); err != nil {
        return "", errors.New("invalid MAC address format")
    }
    // Format as lowercase colon-separated
    clean = strings.ToLower(clean)
    return fmt.Sprintf("%s:%s:%s:%s:%s:%s",
        clean[0:2], clean[2:4], clean[4:6],
        clean[6:8], clean[8:10], clean[10:12]), nil
}
```

#### Serial Normalization

```go
// normalizeSerial converts serial to uppercase alphanumeric only
// Accepts: ST-1234-5678, st12345678, ST 1234 5678
// Returns: ST12345678
func normalizeSerial(input string) (string, error) {
    // Remove all non-alphanumeric characters
    var clean strings.Builder
    for _, r := range strings.TrimSpace(input) {
        if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') {
            clean.WriteRune(r)
        }
    }
    result := strings.ToUpper(clean.String())
    if len(result) < 4 {
        return "", errors.New("serial number too short")
    }
    return result, nil
}
```

### Phase 3: IP Lookup from dhcp_name_cache

#### Configurable Freshness Window

```go
// DHCPCacheFreshnessWindow controls how recent the dhcp_name_cache entry must be
// Default: 5 minutes. Override via MANUAL_CLAIM_FRESHNESS_MINUTES env var.
var DHCPCacheFreshnessWindow = 5 * time.Minute
```

#### IP Lookup Query

```go
// lookupIPByMAC finds the most recent IP for a MAC address
func (h *ClaimHandler) lookupIPByMAC(ctx context.Context, normalizedMAC string) (string, string, error) {
    // Format interval string for PostgreSQL
    intervalStr := fmt.Sprintf("%d minutes", int(DHCPCacheFreshnessWindow.Minutes()))

    var ip, hostname string
    err := h.pool.QueryRow(ctx, `
        SELECT host(ip_address), COALESCE(hostname, '')
        FROM dhcp_name_cache
        WHERE mac_address = $1::macaddr
          AND last_seen_at > NOW() - $2::interval
        ORDER BY last_seen_at DESC
        LIMIT 1
    `, normalizedMAC, intervalStr).Scan(&ip, &hostname)

    if err == pgx.ErrNoRows {
        return "", "", &claimError{code: "device_not_found", msg: "Device not found on network"}
    }
    if err != nil {
        return "", "", fmt.Errorf("lookup IP by MAC: %w", err)
    }
    return ip, hostname, nil
}
```

**Note:** `dhcp_name_cache.mac_address` is stored as PostgreSQL `MACADDR` type (confirmed from migration), so the query casts correctly.

#### LAN Range Validation

```go
var allowedRanges = []string{
    "10.0.0.0/8",
    "172.16.0.0/12",
    "192.168.0.0/16",
}

func isAllowedIP(ipStr string) bool {
    ip := net.ParseIP(ipStr)
    if ip == nil {
        return false
    }
    for _, cidr := range allowedRanges {
        _, network, _ := net.ParseCIDR(cidr)
        if network.Contains(ip) {
            return true
        }
    }
    return false
}
```

### Phase 4: Device Identification (Parallel Probes)

#### Strategy: Parallel Probes with Context Cancellation

Launch parallel probes with strict timeouts. First successful auth+identify wins. All probes must respect context cancellation.

```go
const (
    probeConnectTimeout = 3 * time.Second
    probeRequestTimeout = 5 * time.Second
    probeOverallTimeout = 12 * time.Second
)

type ProbeResult struct {
    DeviceType      string
    Model           string
    SerialNumber    string
    Hostname        string
    FirmwareVersion string
    APIPort         int
    APIProtocol     string
    ModelID         string
    DeviceTypeID    int
}

type probeOutcome struct {
    result   *ProbeResult
    authFail bool  // True if auth specifically failed (401/403 with device-specific response)
    err      error
}

func (p *DeviceIdentifier) Identify(ctx context.Context, ip string) (*ProbeResult, error) {
    ctx, cancel := context.WithTimeout(ctx, probeOverallTimeout)
    defer cancel()

    outcomes := make(chan probeOutcome, 4)

    // Launch probes - each respects ctx.Done()
    go p.probeWithContext(ctx, ip, "ap", outcomes)
    go p.probeWithContext(ctx, ip, "switch", outcomes)
    go p.probeWithContext(ctx, ip, "router", outcomes)
    go p.probeWithContext(ctx, ip, "wattbox", outcomes)

    // Collect results
    var authFailCount int
    var lastErr error
    for i := 0; i < 4; i++ {
        select {
        case outcome := <-outcomes:
            if outcome.result != nil {
                cancel() // Signal other probes to stop
                return outcome.result, nil
            }
            if outcome.authFail {
                authFailCount++
            }
            if outcome.err != nil {
                lastErr = outcome.err
            }
        case <-ctx.Done():
            return nil, &claimError{code: "device_unreachable", msg: "Device identification timed out"}
        }
    }

    // If multiple probes got auth failures, credentials are wrong
    // (single auth fail could be false positive from generic middleware)
    if authFailCount >= 2 {
        return nil, &claimError{code: "auth_failed", msg: "Authentication failed"}
    }
    if lastErr != nil {
        return nil, lastErr
    }
    return nil, &claimError{code: "probe_failed", msg: "Could not identify device type"}
}

// probeWithContext runs a single probe and cancels on context done
func (p *DeviceIdentifier) probeWithContext(ctx context.Context, ip, deviceType string, out chan<- probeOutcome) {
    result, authFail, err := p.probeSingle(ctx, ip, deviceType)
    select {
    case out <- probeOutcome{result: result, authFail: authFail, err: err}:
    case <-ctx.Done():
        // Context cancelled, don't block
    }
}
```

#### Auth Failure Detection

Only treat 401/403 as `authFail=true` when the response body contains device-type-specific content:

| Device Type | Port | Auth Fail Indicators |
|-------------|------|---------------------|
| Access Point | 4431 | 401 + response contains "token" or "jwt" |
| Switch | 443 | 401 + response contains "Unauthorized" + path starts with /api/v1 |
| Router | 8443 | 401 + response contains "cgi-bin" |
| WattBox | 23 | Telnet login failed with "login:" prompt |

This prevents false positives from generic reverse proxies or middleware.

### Phase 5: Race-Safe Claiming

#### Correct Transaction Logic

Use `FOR UPDATE NOWAIT` to get a distinguishable error when locked, with unique constraint as fallback.

```go
// claimError is a typed error for claim-specific failures
type claimError struct {
    code string
    msg  string
}

func (e *claimError) Error() string { return e.msg }

func (h *ClaimHandler) manualClaimDevice(ctx context.Context, tx pgx.Tx, normalizedMAC string, result *ProbeResult) (string, error) {
    // Step 1: Try to lock existing row
    var existingID string
    var isClaimed bool
    err := tx.QueryRow(ctx, `
        SELECT id::text, is_claimed
        FROM devices
        WHERE mac_address = $1::macaddr
        FOR UPDATE NOWAIT
    `, normalizedMAC).Scan(&existingID, &isClaimed)

    if err == nil {
        // Row exists and we locked it
        if isClaimed {
            return "", &claimError{code: "already_claimed", msg: "Device is already claimed"}
        }
        // Update existing unclaimed device to claimed
        return h.updateDeviceToClaimed(ctx, tx, existingID, result)
    }

    // Check error type
    var pgErr *pgconn.PgError
    if errors.As(err, &pgErr) && pgErr.Code == "55P03" {
        // 55P03 = lock_not_available (NOWAIT couldn't get lock)
        return "", &claimError{code: "claim_in_progress", msg: "Another claim is in progress"}
    }

    if err != pgx.ErrNoRows {
        return "", fmt.Errorf("check existing device: %w", err)
    }

    // Step 2: Row doesn't exist, insert new claimed device
    deviceID, insertErr := h.insertClaimedDevice(ctx, tx, normalizedMAC, result)
    if insertErr != nil {
        // Check for unique constraint violation (race condition)
        var pgErr *pgconn.PgError
        if errors.As(insertErr, &pgErr) && pgErr.Code == "23505" {
            // 23505 = unique_violation
            // Someone else inserted between our check and insert
            return "", &claimError{code: "claim_in_progress", msg: "Another claim completed first"}
        }
        return "", insertErr
    }

    return deviceID, nil
}
```

#### Error Mapping in Handler

```go
func (h *ClaimHandler) ManualClaim(w http.ResponseWriter, r *http.Request) {
    // ... validation, lookup, identify ...

    // Map errors to HTTP responses
    var ce *claimError
    if errors.As(err, &ce) {
        switch ce.code {
        case "invalid_mac_format", "invalid_serial_format":
            writeJSONError(w, http.StatusBadRequest, ce.code, ce.msg)
        case "device_not_found":
            writeJSONError(w, http.StatusNotFound, ce.code, ce.msg)
        case "ip_not_allowed":
            writeJSONError(w, http.StatusForbidden, ce.code, ce.msg)
        case "auth_failed":
            writeJSONError(w, http.StatusUnauthorized, ce.code, ce.msg)
        case "already_claimed", "claim_in_progress":
            writeJSONError(w, http.StatusConflict, ce.code, ce.msg)
        case "serial_mismatch":
            writeJSONError(w, http.StatusUnprocessableEntity, ce.code, ce.msg)
        case "device_unreachable":
            writeJSONError(w, http.StatusServiceUnavailable, ce.code, ce.msg)
        case "probe_failed":
            writeJSONError(w, http.StatusInternalServerError, ce.code, ce.msg)
        default:
            writeJSONError(w, http.StatusInternalServerError, "error", ce.msg)
        }
        return
    }
    // Generic error
    writeJSONError(w, http.StatusInternalServerError, "error", err.Error())
}
```

### Phase 6: Frontend Implementation

#### ManualClaimModal Component

Location: `frontend/src/components/ManualClaimModal.tsx`

Features:
- MAC Address input with format hint
- Serial Number input
- Loading state with progress indicator
- Graceful handling of `claim_in_progress` (show "still working, please wait" and auto-retry)
- Error display mapping backend codes to user-friendly messages
- Success state showing claimed device details

```typescript
interface ManualClaimModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSuccess: (device: ClaimedDevice) => void;
}

// Error message mapping
const errorMessages: Record<string, { title: string; message: string; retryable: boolean }> = {
  invalid_mac_format: {
    title: "Invalid MAC Address",
    message: "Please enter a valid MAC address (e.g., 4C:13:65:11:08:00)",
    retryable: false,
  },
  device_not_found: {
    title: "Device Not Found",
    message: "Device not found on network. Make sure it's powered on and connected.",
    retryable: true,
  },
  auth_failed: {
    title: "Authentication Failed",
    message: "Could not log in to device. Check credentials in Settings.",
    retryable: false,
  },
  already_claimed: {
    title: "Already Claimed",
    message: "This device is already claimed.",
    retryable: false,
  },
  claim_in_progress: {
    title: "Claim In Progress",
    message: "Another claim is in progress. Waiting...",
    retryable: true,  // Auto-retry after 2s
  },
  serial_mismatch: {
    title: "Serial Mismatch",
    message: "Serial number doesn't match the device. Check the label on the device.",
    retryable: false,
  },
  device_unreachable: {
    title: "Device Unreachable",
    message: "Could not connect to device. Check if it's powered on.",
    retryable: true,
  },
};
```

#### Auto-Retry Logic for `claim_in_progress`

```typescript
const handleSubmit = async () => {
  setLoading(true);
  setError(null);

  let attempts = 0;
  const maxAttempts = 3;

  while (attempts < maxAttempts) {
    try {
      const result = await httpClient.manualClaimDevice(mac, serial);
      onSuccess(result);
      return;
    } catch (err) {
      const code = err.response?.data?.code;
      if (code === 'claim_in_progress' && attempts < maxAttempts - 1) {
        attempts++;
        setError({ title: 'Claim In Progress', message: `Waiting... (attempt ${attempts + 1}/${maxAttempts})` });
        await new Promise(r => setTimeout(r, 2000));
        continue;
      }
      setError(errorMessages[code] || { title: 'Error', message: err.message, retryable: false });
      break;
    }
  }
  setLoading(false);
};
```

---

## Files to Create/Modify

### New Files

| File | Purpose |
|------|---------|
| `backend/internal/discovery/identifier.go` | Device type identification via parallel probes |
| `frontend/src/components/ManualClaimModal.tsx` | Manual claim UI modal |

### Modified Files

| File | Changes |
|------|---------|
| `backend/internal/handlers/claim.go` | Add `ManualClaim` handler method, normalization functions, claimError type |
| `backend/cmd/server/main.go` | Register `/api/v1/devices/manual-claim` route |
| `frontend/src/app/devices/page.tsx` | Add manual claim button & modal |
| `frontend/src/lib/httpClient.ts` | Add `manualClaimDevice` API method |

---

## Implementation Order

1. Add `claimError` type and normalization functions to `claim.go`
2. Add IP lookup function with configurable freshness
3. Create `identifier.go` with parallel probe logic + context cancellation
4. Add `ManualClaim` method to `claim.go` with error mapping and race-safe claiming
5. Register route in `main.go`
6. Add `manualClaimDevice` to httpClient
7. Create `ManualClaimModal` component with auto-retry
8. Update Devices page with button and modal
9. Test end-to-end with AN-530 wireless bridge

---

## Security Summary

| Concern | Mitigation |
|---------|------------|
| Wrong device claimed | Normalized serial verification required |
| MAC typo claims wrong device | Serial verification catches this |
| Probing arbitrary IPs | LAN range validation (RFC1918 only) |
| Race conditions | FOR UPDATE NOWAIT + unique constraint fallback |
| Stale IP → wrong device | Configurable freshness window (default 5m) |
| Credential exposure | Use same encrypted credential flow as existing claim |
| Probe goroutine leaks | All probes respect context cancellation |

---

## Testing Plan

### Unit Tests

1. MAC normalization (various formats: colons, dashes, no separator, mixed case)
2. Serial normalization (hyphens, spaces, mixed case)
3. LAN range validation (valid/invalid IPs)
4. Error code mapping (claimError → HTTP status)

### Integration Tests

1. Full flow: MAC lookup → probe → verify serial → claim
2. Race condition: concurrent claims for same MAC (expect one succeeds, one gets claim_in_progress)
3. Already claimed device
4. Serial mismatch (normalized comparison)
5. Device unreachable (timeout handling)
6. Auth failed with correct device type detection

### Manual Testing

1. Claim AN-530 wireless bridge AP (original use case)
2. Verify error messages are clear and actionable
3. Test with various MAC/Serial input formats
4. Test auto-retry on claim_in_progress
