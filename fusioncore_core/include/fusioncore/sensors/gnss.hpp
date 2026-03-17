#pragma once

#include "fusioncore/state.hpp"
#include <Eigen/Dense>
#include <cmath>

namespace fusioncore {
namespace sensors {

// ─── GNSS position measurement (3-dimensional) ──────────────────────────────
// [x, y, z] in ENU frame (converted from ECEF or lat/lon/alt)

constexpr int GNSS_POS_DIM = 3;

using GnssPosMeasurement = Eigen::Matrix<double, GNSS_POS_DIM, 1>;
using GnssPosNoiseMatrix = Eigen::Matrix<double, GNSS_POS_DIM, GNSS_POS_DIM>;

// ─── GNSS heading measurement (1-dimensional) ───────────────────────────────
// Dual antenna heading — direct yaw measurement

constexpr int GNSS_HDG_DIM = 1;

using GnssHdgMeasurement = Eigen::Matrix<double, GNSS_HDG_DIM, 1>;
using GnssHdgNoiseMatrix = Eigen::Matrix<double, GNSS_HDG_DIM, GNSS_HDG_DIM>;

// ─── GNSS quality parameters ─────────────────────────────────────────────────

struct GnssParams {
  // Base position noise (m) — scaled by HDOP/VDOP
  double base_noise_xy = 1.0;   // horizontal (m)
  double base_noise_z  = 2.0;   // vertical (m) — always worse than horizontal

  // Dual antenna heading noise (rad)
  double heading_noise = 0.02;  // ~1 degree

  // Quality thresholds — reject measurements below these
  double max_hdop = 4.0;        // reject if HDOP worse than this
  double max_vdop = 6.0;        // reject if VDOP worse than this
  int    min_satellites = 4;    // reject if fewer satellites
};

// ─── GNSS fix quality ────────────────────────────────────────────────────────

enum class GnssFixType {
  NO_FIX    = 0,
  GPS_FIX   = 1,
  DGPS_FIX  = 2,
  RTK_FLOAT = 3,
  RTK_FIXED = 4
};

struct GnssFix {
  double x = 0.0;   // ENU position (meters)
  double y = 0.0;
  double z = 0.0;

  double hdop = 99.0;  // horizontal dilution of precision
  double vdop = 99.0;  // vertical dilution of precision

  int    satellites  = 0;
  GnssFixType fix_type = GnssFixType::NO_FIX;

  bool is_valid(const GnssParams& p) const {
    return fix_type != GnssFixType::NO_FIX
        && hdop <= p.max_hdop
        && vdop <= p.max_vdop
        && satellites >= p.min_satellites;
  }
};

struct GnssHeading {
  double heading_rad = 0.0;  // yaw in ENU frame (rad)
  double accuracy_rad = 0.1; // reported accuracy
  bool   valid = false;
};

// ─── Measurement functions ───────────────────────────────────────────────────

// h(x): state -> expected GNSS position measurement
inline GnssPosMeasurement gnss_pos_measurement_function(const StateVector& x) {
  GnssPosMeasurement z;
  z[0] = x[X];
  z[1] = x[Y];
  z[2] = x[Z];
  return z;
}

// h(x): state -> expected GNSS heading measurement
inline GnssHdgMeasurement gnss_hdg_measurement_function(const StateVector& x) {
  GnssHdgMeasurement z;
  z[0] = x[YAW];
  return z;
}

// ─── Quality-aware noise matrix ───────────────────────────────────────────────
// This is the key differentiator — we scale noise by HDOP/VDOP
// A fix with HDOP=1.0 gets tight noise. HDOP=3.5 gets loose noise.
// robot_localization uses fixed covariance — FusionCore adapts.

inline GnssPosNoiseMatrix gnss_pos_noise_matrix(
  const GnssParams& p,
  const GnssFix& fix
) {
  GnssPosNoiseMatrix R = GnssPosNoiseMatrix::Zero();

  // Scale base noise by DOP values
  double sigma_xy = p.base_noise_xy * fix.hdop;
  double sigma_z  = p.base_noise_z  * fix.vdop;

  R(0,0) = sigma_xy * sigma_xy;
  R(1,1) = sigma_xy * sigma_xy;
  R(2,2) = sigma_z  * sigma_z;

  return R;
}

inline GnssHdgNoiseMatrix gnss_hdg_noise_matrix(
  const GnssParams& p,
  const GnssHeading& hdg
) {
  GnssHdgNoiseMatrix R;
  double sigma = std::max(p.heading_noise, hdg.accuracy_rad);
  R(0,0) = sigma * sigma;
  return R;
}

// ─── ECEF to ENU conversion ───────────────────────────────────────────────────
// This is why FusionCore uses ECEF — no singularities, no UTM zone boundaries

struct ECEFPoint {
  double x, y, z;  // meters
};

struct LLAPoint {
  double lat_rad;  // latitude (radians)
  double lon_rad;  // longitude (radians)
  double alt_m;    // altitude (meters)
};

// Convert ECEF to ENU relative to a reference point
inline Eigen::Vector3d ecef_to_enu(
  const ECEFPoint& point,
  const ECEFPoint& ref,
  const LLAPoint&  ref_lla
) {
  double dx = point.x - ref.x;
  double dy = point.y - ref.y;
  double dz = point.z - ref.z;

  double sin_lat = std::sin(ref_lla.lat_rad);
  double cos_lat = std::cos(ref_lla.lat_rad);
  double sin_lon = std::sin(ref_lla.lon_rad);
  double cos_lon = std::cos(ref_lla.lon_rad);

  // ENU rotation matrix
  double e = -sin_lon*dx           + cos_lon*dy;
  double n = -sin_lat*cos_lon*dx   - sin_lat*sin_lon*dy + cos_lat*dz;
  double u =  cos_lat*cos_lon*dx   + cos_lat*sin_lon*dy + sin_lat*dz;

  return Eigen::Vector3d(e, n, u);
}

} // namespace sensors
} // namespace fusioncore
