#pragma once

#include <flexi_cfg/config/reader.h>
#include <flexi_cfg/utils.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <array>

#include "data_registrar.h"

namespace control::VMC {

struct Gains {
 private:
  struct CartesianGains {
   protected:
    friend Gains;
    std::array<double, 3> Kp_ = {0};
    std::array<double, 3> Kd_ = {0};

   public:
    Eigen::Map<Eigen::Vector3d> Kp{Kp_.data()};
    Eigen::Map<Eigen::Vector3d> Kd{Kd_.data()};
  };

 public:
  CartesianGains linear;
  CartesianGains angular;

  /// @brief Read gains from the configuration file
  void read(const ConfigReader& cfg, const std::string& prefix) {
    using utils::makeName;
    cfg.getValue(makeName(prefix, "vmc.linear.Kp"), linear.Kp_);
    cfg.getValue(makeName(prefix, "vmc.linear.Kd"), linear.Kd_);
    cfg.getValue(makeName(prefix, "vmc.angular.Kp"), angular.Kp_);
    cfg.getValue(makeName(prefix, "vmc.angular.Kd"), angular.Kd_);
  }

  /// @brief Register the gains with the data registrar
  void registerVars(const std::string& prefix, DataRegistrar& dr) {
    using utils::makeName;
    dr.registerVar(makeName(prefix, "vmc.linear.Kp"), &linear.Kp_, {log_helpers::cartesian});
    dr.registerVar(makeName(prefix, "vmc.linear.Kd"), &linear.Kd_, {log_helpers::cartesian});
    dr.registerVar(makeName(prefix, "vmc.angular.Kp"), &angular.Kp_, {log_helpers::cartesian});
    dr.registerVar(makeName(prefix, "vmc.angular.Kd"), &angular.Kd_, {log_helpers::cartesian});
  }
};

struct Input {
  // Linear inputs
  Eigen::Vector3d pos = Eigen::Vector3d::Zero();
  Eigen::Vector3d pos_d = Eigen::Vector3d::Zero();
  Eigen::Vector3d vel = Eigen::Vector3d::Zero();
  Eigen::Vector3d vel_d = Eigen::Vector3d::Zero();

  // Angular inputs
  Eigen::Quaterniond quat = Eigen::Quaterniond::Identity();
  Eigen::Quaterniond quat_d = Eigen::Quaterniond::Identity();
  Eigen::Vector3d omega = Eigen::Vector3d::Zero();
  Eigen::Vector3d omega_d = Eigen::Vector3d::Zero();

  // Feedforward force & moment
  Eigen::Vector3d force_ff = Eigen::Vector3d::Zero();
  Eigen::Vector3d moment_ff = Eigen::Vector3d::Zero();

  /// @brief Register the measured and desired values with the registrar
  void registerVars(const std::string& prefix, DataRegistrar& dr) {
    using utils::makeName;
    dr.registerVar(makeName(prefix, "vmc.pos"), &pos, {log_helpers::cartesian});
    dr.registerVar(makeName(prefix, "vmc.pos_d"), &pos_d, {log_helpers::cartesian});
    dr.registerVar(makeName(prefix, "vmc.vel"), &vel, {log_helpers::cartesian});
    dr.registerVar(makeName(prefix, "vmc.vel_d"), &vel_d, {log_helpers::cartesian});
    dr.registerVar(makeName(prefix, "vmc.F_ff"), &force_ff, {log_helpers::cartesian});

    dr.registerVar(makeName(prefix, "vmc.quat"), &quat);
    dr.registerVar(makeName(prefix, "vmc.quat_d"), &quat_d);
    dr.registerVar(makeName(prefix, "vmc.omega"), &omega, {log_helpers::cartesian});
    dr.registerVar(makeName(prefix, "vmc.omega_d"), &omega_d, {log_helpers::cartesian});
    dr.registerVar(makeName(prefix, "vmc.M_ff"), &moment_ff, {log_helpers::cartesian});
  }
};

/// @brief Compute the PD feedback between the provided error terms
/// @param err_pos The proportional error
/// @param err_vel The differential error
/// @param Kp The proportional gain
/// @param Kd The derivative gain
///
/// @return The cartesian feedback value
inline Eigen::Vector3d computeFeedbackFromError(const Eigen::Vector3d& err_pos,
                                                const Eigen::Vector3d& err_vel,
                                                const Eigen::Vector3d& Kp,
                                                const Eigen::Vector3d& Kd) {
  return Kp.cwiseProduct(err_pos) + Kd.cwiseProduct(err_vel);
}

/// @brief Compute the translational feedback about a desired position & velocity
/// @param pos_d The desired position
/// @param pos   The measured position
/// @param vel_d The desired velocity
/// @param vel   The measured velocity
/// @param Kp The proportional gain
/// @param Kd The derivative gain
///
/// @return The translational feedback value
inline Eigen::Vector3d computeLinearFeedback(const Eigen::Vector3d& pos_d,
                                             const Eigen::Vector3d& pos,
                                             const Eigen::Vector3d& vel_d,
                                             const Eigen::Vector3d& vel, const Eigen::Vector3d& Kp,
                                             const Eigen::Vector3d& Kd) {
  return computeFeedbackFromError(pos_d - pos, vel_d - vel, Kp, Kd);
}

/// @brief Compute the angular feedback about a desired position & velocity
/// @param quaternion_d The desired orientation of the system expressed as a quaternion
/// @param quaternion   The measured orientation of the system expressed as a quaternion
/// @param omega_d The desired angular velocity of the system
/// @param omega   The measured angular velocity of the system
/// @param Kp The proportional gain
/// @param Kd The derivative gain
///
/// @note The orientation and angular velocities must be expressed in the same frame
///
/// @return The angular feedback value
inline Eigen::Vector3d computeAngularFeedback(const Eigen::Quaternion<double>& quaternion_d,
                                              const Eigen::Quaternion<double>& quaternion,
                                              const Eigen::Vector3d& omega_d,
                                              const Eigen::Vector3d& omega,
                                              const Eigen::Vector3d& Kp,
                                              const Eigen::Vector3d& Kd) {
  // Calculate the error quaternion
  Eigen::Quaterniond q_error = quaternion_d * quaternion.conjugate();
  if (q_error.w() < 0) {
    // If the scalar component is less than zero, then the error between the two rotations is larger
    // than pi, so the nearest rotation is the conjugate of the error.
    q_error = q_error.conjugate();
  }

  const Eigen::Vector3d omega_error = omega_d - omega;
  return computeFeedbackFromError(q_error.vec(), omega_error, Kp, Kd);
}

}  // namespace control::VMC
