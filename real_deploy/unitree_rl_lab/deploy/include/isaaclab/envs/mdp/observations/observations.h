// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include "isaaclab/envs/manager_based_rl_env.h"

namespace isaaclab
{
namespace mdp
{

REGISTER_OBSERVATION(base_ang_vel)
{
    auto & asset = env->robot;
    auto & data = asset->data.root_ang_vel_b;
    return std::vector<float>(data.data(), data.data() + data.size());
}

REGISTER_OBSERVATION(projected_gravity)
{
    auto & asset = env->robot;
    auto & data = asset->data.projected_gravity_b;
    return std::vector<float>(data.data(), data.data() + data.size());
}

REGISTER_OBSERVATION(joint_pos)
{
    auto & asset = env->robot;
    std::vector<float> data;

    std::vector<int> joint_ids;
    try {
        joint_ids = params["asset_cfg"]["joint_ids"].as<std::vector<int>>();
    } catch(const std::exception& e) {
    }

    if(joint_ids.empty())
    {
        data.resize(asset->data.joint_pos.size());
        for(size_t i = 0; i < asset->data.joint_pos.size(); ++i)
        {
            data[i] = asset->data.joint_pos[i];
        }
    }
    else
    {
        data.resize(joint_ids.size());
        for(size_t i = 0; i < joint_ids.size(); ++i)
        {
            data[i] = asset->data.joint_pos[joint_ids[i]];
        }
    }

    return data;
}

REGISTER_OBSERVATION(joint_pos_rel)
{
    auto & asset = env->robot;
    std::vector<float> data;

    data.resize(asset->data.joint_pos.size());
    for(size_t i = 0; i < asset->data.joint_pos.size(); ++i) {
        data[i] = asset->data.joint_pos[i] - asset->data.default_joint_pos[i];
    }

    try {
        std::vector<int> joint_ids;
        joint_ids = params["asset_cfg"]["joint_ids"].as<std::vector<int>>();
        if(!joint_ids.empty()) {
            std::vector<float> tmp_data;
            tmp_data.resize(joint_ids.size());
            for(size_t i = 0; i < joint_ids.size(); ++i){
                tmp_data[i] = data[joint_ids[i]];
            }
            data = tmp_data;
        }
    } catch(const std::exception& e) {
    
    }

    return data;
}

REGISTER_OBSERVATION(joint_vel_rel)
{
    auto & asset = env->robot;
    auto data = asset->data.joint_vel;

    try {
        const std::vector<int> joint_ids = params["asset_cfg"]["joint_ids"].as<std::vector<int>>();

        if(!joint_ids.empty()) {
            data.resize(joint_ids.size());
            for(size_t i = 0; i < joint_ids.size(); ++i) {
                data[i] = asset->data.joint_vel[joint_ids[i]];
            }
        }
    } catch(const std::exception& e) {
    }
    return std::vector<float>(data.data(), data.data() + data.size());
}

REGISTER_OBSERVATION(last_action)
{
    auto data = env->action_manager->action();
    return std::vector<float>(data.data(), data.data() + data.size());
};

REGISTER_OBSERVATION(velocity_commands)
{
    std::vector<float> obs(3);
    auto & asset = env->robot;

    const auto cfg = env->cfg["commands"]["base_velocity"]["ranges"];

    if (asset->data.use_velocity_command_override) {
        const auto & command = asset->data.velocity_command_override;
        obs[0] = std::clamp(command[0], cfg["lin_vel_x"][0].as<float>(), cfg["lin_vel_x"][1].as<float>());
        obs[1] = std::clamp(command[1], cfg["lin_vel_y"][0].as<float>(), cfg["lin_vel_y"][1].as<float>());
        obs[2] = std::clamp(command[2], cfg["ang_vel_z"][0].as<float>(), cfg["ang_vel_z"][1].as<float>());
        return obs;
    }

    auto & joystick = asset->data.joystick;

    obs[0] = std::clamp(joystick->ly(), cfg["lin_vel_x"][0].as<float>(), cfg["lin_vel_x"][1].as<float>());
    obs[1] = std::clamp(-joystick->lx(), cfg["lin_vel_y"][0].as<float>(), cfg["lin_vel_y"][1].as<float>());
    obs[2] = std::clamp(-joystick->rx(), cfg["ang_vel_z"][0].as<float>(), cfg["ang_vel_z"][1].as<float>());

    return obs;
}

REGISTER_OBSERVATION(gait_phase)
{
    float period = params["period"].as<float>();
    float delta_phase = env->step_dt * (1.0f / period);

    env->global_phase += delta_phase;
    env->global_phase = std::fmod(env->global_phase, 1.0f);

    std::vector<float> obs(2);
    obs[0] = std::sin(env->global_phase * 2 * M_PI);
    obs[1] = std::cos(env->global_phase * 2 * M_PI);
    return obs;
}

inline float wrap_to_pi(float angle)
{
    while (angle > static_cast<float>(M_PI)) {
        angle -= static_cast<float>(2.0 * M_PI);
    }
    while (angle < static_cast<float>(-M_PI)) {
        angle += static_cast<float>(2.0 * M_PI);
    }
    return angle;
}

inline std::array<float, 3> euler_xyz_from_quat(const Eigen::Quaternionf& quat)
{
    const auto rot = quat.normalized().toRotationMatrix();
    const float roll = std::atan2(rot(2, 1), rot(2, 2));
    const float pitch = std::asin(std::clamp(-rot(2, 0), -1.0f, 1.0f));
    const float yaw = std::atan2(rot(1, 0), rot(0, 0));
    return {wrap_to_pi(roll), wrap_to_pi(pitch), wrap_to_pi(yaw)};
}

inline float parkour_command_x_target(ManagerBasedRLEnv* env)
{
    auto & asset = env->robot;
    float command_x = 0.0f;
    if (asset->data.use_velocity_command_override) {
        command_x = asset->data.velocity_command_override[0];
    } else if (asset->data.joystick != nullptr) {
        command_x = asset->data.joystick->ly();
    }

    float min_x = 0.0f;
    float max_x = 0.35f;
    const auto command_cfg = env->cfg["commands"]["base_velocity"]["ranges"]["lin_vel_x"];
    if (command_cfg && command_cfg.IsSequence() && command_cfg.size() >= 2) {
        min_x = command_cfg[0].as<float>();
        max_x = command_cfg[1].as<float>();
    }
    return std::clamp(command_x, min_x, max_x);
}

REGISTER_OBSERVATION(parkour_proprio)
{
    auto & asset = env->robot;
    std::vector<float> obs;
    obs.reserve(53);

    for (int i = 0; i < 3; ++i) {
        obs.push_back(asset->data.root_ang_vel_b[i] * 0.25f);
    }

    const auto euler = euler_xyz_from_quat(asset->data.root_quat_w);
    obs.push_back(euler[0]);
    obs.push_back(euler[1]);

    const float yaw = euler[2];
    obs.push_back(0.0f);
    obs.push_back(yaw);
    obs.push_back(yaw);

    obs.push_back(0.0f);
    obs.push_back(0.0f);

    const float target_x = parkour_command_x_target(env);
    const float ramp_rate = env->cfg["parkour"]["command_ramp_rate"].as<float>(0.35f);
    const float max_delta = std::max(0.0f, ramp_rate * env->step_dt);
    const float delta = std::clamp(target_x - asset->data.parkour_command_x, -max_delta, max_delta);
    asset->data.parkour_command_x += delta;
    obs.push_back(asset->data.parkour_command_x);

    obs.push_back(1.0f);
    obs.push_back(0.0f);

    for (int i = 0; i < asset->data.joint_pos.size(); ++i) {
        obs.push_back(asset->data.joint_pos[i] - asset->data.default_joint_pos[i]);
    }

    for (int i = 0; i < asset->data.joint_vel.size(); ++i) {
        obs.push_back(asset->data.joint_vel[i] * 0.05f);
    }

    auto last_action_data = env->action_manager->action();
    obs.insert(obs.end(), last_action_data.begin(), last_action_data.end());

    const float contact_threshold = env_float(
        "FOOT_FORCE_THRESHOLD",
        env->cfg["parkour"]["foot_force_threshold"].as<float>(2.0f)
    );
    const std::array<int, 4> foot_force_unitree_to_isaac = {1, 0, 3, 2}; // Unitree FR,FL,RR,RL -> Isaac FL,FR,RL,RR
    for (const int idx : foot_force_unitree_to_isaac) {
        const bool current_contact = asset->data.foot_force[idx] > contact_threshold;
        const bool previous_contact = asset->data.foot_contact_prev[idx] > 0.5f;
        const bool contact = current_contact || previous_contact;
        asset->data.foot_contact_prev[idx] = current_contact ? 1.0f : 0.0f;
        obs.push_back((contact ? 1.0f : 0.0f) - 0.5f);
    }

    return obs;
}

REGISTER_OBSERVATION(parkour_raw_foot_force)
{
    auto & asset = env->robot;
    return std::vector<float>(
        asset->data.foot_force.begin(),
        asset->data.foot_force.end()
    );
}

REGISTER_OBSERVATION(parkour_raw_foot_force_est)
{
    auto & asset = env->robot;
    return std::vector<float>(
        asset->data.foot_force_est.begin(),
        asset->data.foot_force_est.end()
    );
}

}
}
