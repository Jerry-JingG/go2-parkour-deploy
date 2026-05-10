// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "isaaclab/assets/articulation/articulation.h"

namespace unitree
{

template <typename LowStatePtr>
class BaseArticulation : public isaaclab::Articulation
{
public:
    BaseArticulation(LowStatePtr lowstate_)
    : lowstate(lowstate_)
    {
        data.joystick = &lowstate->joystick;
    }

    void update() override
    {
        std::lock_guard<std::mutex> lock(lowstate->mutex_);
        // base_angular_velocity
        for(int i(0); i<3; i++) {
            const float gyro = lowstate->msg_.imu_state().gyroscope()[i];
            data.root_ang_vel_b[i] = gyro;
            data.sdk_imu_gyro[i] = gyro;
        }
        // project_gravity_body
        data.root_quat_w = Eigen::Quaternionf(
            lowstate->msg_.imu_state().quaternion()[0],
            lowstate->msg_.imu_state().quaternion()[1],
            lowstate->msg_.imu_state().quaternion()[2],
            lowstate->msg_.imu_state().quaternion()[3]
        );
        for (int i(0); i < 4; ++i) {
            data.sdk_imu_quat_wxyz[i] = lowstate->msg_.imu_state().quaternion()[i];
        }
        data.projected_gravity_b = data.root_quat_w.conjugate() * data.GRAVITY_VEC_W;
        for (size_t i(0); i < data.sdk_joint_pos.size(); ++i) {
            data.sdk_joint_pos[i] = lowstate->msg_.motor_state()[i].q();
            data.sdk_joint_vel[i] = lowstate->msg_.motor_state()[i].dq();
        }
        // joint positions and velocities
        for(int i(0); i< data.joint_ids_map.size(); i++) {
            data.joint_pos[i] = lowstate->msg_.motor_state()[data.joint_ids_map[i]].q();
            data.joint_vel[i] = lowstate->msg_.motor_state()[data.joint_ids_map[i]].dq();
        }

        const auto & foot_force = lowstate->msg_.foot_force();
        const auto & foot_force_est = lowstate->msg_.foot_force_est();
        for (int i(0); i < data.foot_force.size(); ++i) {
            data.foot_force[i] = static_cast<float>(foot_force[i]);
            data.sdk_foot_force[i] = static_cast<float>(foot_force[i]);
            data.foot_force_est[i] = static_cast<float>(foot_force_est[i]);
            data.sdk_foot_force_est[i] = static_cast<float>(foot_force_est[i]);
        }
    }

    LowStatePtr lowstate;
};

}
