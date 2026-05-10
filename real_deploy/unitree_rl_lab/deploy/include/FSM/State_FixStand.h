// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "FSMState.h"
#include "LinearInterpolator.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "isaaclab/envs/mdp/observations/observations.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <memory>
#include <spdlog/spdlog.h>
#include <string>
#include <thread>

class State_FixStand : public FSMState
{
public:
    State_FixStand(int state, std::string state_string = "FixStand") 
    : FSMState(state, state_string) 
    {
        ts_ = param::config["FSM"]["FixStand"]["ts"].as<std::vector<float>>();
        qs_ = param::config["FSM"]["FixStand"]["qs"].as<std::vector<std::vector<float>>>();
        assert(ts_.size() == qs_.size());
        stand_record_remote_expr_ = env_string_local(
            "PARKOUR_STAND_RECORD_REMOTE_EXPR",
            "LT + X.on_pressed"
        );
        auto ast = unitree::common::dsl::Parser(stand_record_remote_expr_).Parse();
        stand_record_remote_check_ = unitree::common::dsl::Compile(*ast);

        auto cfg = param::config["FSM"][state_string];
        if (cfg["auto_transition"]) {
            const auto target_fsm = cfg["auto_transition"].as<std::string>();
            const auto after = cfg["auto_transition_after"].as<float>(ts_.back());
            registered_checks.emplace_back(
                std::make_pair(
                    [this, after]()->bool {
                        const float t = (double)unitree::common::GetCurrentTimeMillisecond() * 1e-3 - t0_;
                        return t >= after;
                    },
                    FSMStringMap.right.at(target_fsm)
                )
            );
        }
    }

    ~State_FixStand()
    {
        stop_stand_record_thread();
    }

    void enter()
    {
        // set gain
        static auto kp = param::config["FSM"]["FixStand"]["kp"].as<std::vector<float>>();
        static auto kd = param::config["FSM"]["FixStand"]["kd"].as<std::vector<float>>();
        for(int i(0); i < kp.size(); ++i)
        {
            auto & motor = lowcmd->msg_.motor_cmd()[i];
            motor.kp() = kp[i];
            motor.kd() = kd[i];
            motor.dq() = motor.tau() = 0;
        }


        // set initial position
        std::vector<float> q0;
        for(int i(0); i < kp.size(); ++i) {
            q0.push_back(lowcmd->msg_.motor_cmd()[i].q());
        }
        qs_[0] = q0;
        t0_ = (double)unitree::common::GetCurrentTimeMillisecond() * 1e-3;
    }

    void run()
    {
        float t = (double)unitree::common::GetCurrentTimeMillisecond() * 1e-3 - t0_;
        auto q = linear_interpolate(t, ts_, qs_);
        
        for(int i(0); i < q.size(); ++i) {
            lowcmd->msg_.motor_cmd()[i].q() = q[i];
        }
        handle_stand_record_trigger();
    }

    void exit()
    {
        stop_stand_record_thread();
    }

private:
    double t0_;
    std::vector<float> ts_;
    std::vector<std::vector<float>> qs_;
    std::thread stand_record_thread_;
    std::atomic_bool stand_record_running_{false};
    std::atomic_bool stand_record_stop_{false};
    std::string last_stand_record_key_;
    std::string stand_record_remote_expr_;
    std::function<bool(const unitree::common::UnitreeJoystick&)> stand_record_remote_check_;

    static float env_float_local(const char* name, float fallback)
    {
        const char* value = std::getenv(name);
        if (value == nullptr || value[0] == '\0') {
            return fallback;
        }
        char* end = nullptr;
        const float parsed = std::strtof(value, &end);
        if (end == value) {
            return fallback;
        }
        return parsed;
    }

    static std::string env_string_local(const char* name, const std::string& fallback)
    {
        const char* value = std::getenv(name);
        if (value == nullptr || value[0] == '\0') {
            return fallback;
        }
        return value;
    }

    void handle_stand_record_trigger()
    {
        if (stand_record_remote_check_ && stand_record_remote_check_(FSMState::lowstate->joystick)) {
            start_stand_record_thread();
        }

        if (!FSMState::keyboard_control_enabled() || !FSMState::keyboard) {
            return;
        }
        const auto key = FSMState::keyboard->key();
        if (key.empty()) {
            last_stand_record_key_.clear();
            return;
        }
        if (key == last_stand_record_key_) {
            return;
        }
        last_stand_record_key_ = key;

        const auto record_key = env_string_local("PARKOUR_STAND_RECORD_KEY", "r");
        if (normalize_keyboard_key(key) == normalize_keyboard_key(record_key)) {
            start_stand_record_thread();
        }
    }

    std::unique_ptr<isaaclab::ManagerBasedRLEnv> make_stand_record_env()
    {
        const auto policy_dir_arg = env_string_local(
            "PARKOUR_STAND_RECORD_POLICY_DIR",
            "config/policy/parkour_depth"
        );
        auto policy_dir = param::parser_policy_dir(policy_dir_arg);
        auto deploy_cfg = YAML::LoadFile(policy_dir / "params" / "deploy.yaml");
        if (!deploy_cfg["socket_path"]) {
            throw std::runtime_error("FixStand stand-record requires socket_path in deploy.yaml");
        }

        auto record_env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
            deploy_cfg,
            std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate)
        );
        record_env->alg = std::make_unique<isaaclab::SocketRunner>(
            deploy_cfg["socket_path"].as<std::string>(),
            deploy_cfg["socket_action_dim"].as<int>(record_env->action_manager->total_action_dim()),
            deploy_cfg["socket_input_name"].as<std::string>("obs")
        );
        return record_env;
    }

    void start_stand_record_thread()
    {
        if (stand_record_running_.load()) {
            spdlog::warn("FixStand stand-record is already running.");
            return;
        }
        if (stand_record_thread_.joinable()) {
            stand_record_thread_.join();
        }

        const float seconds = env_float_local("PARKOUR_STAND_RECORD_SECONDS", 5.0f);
        if (seconds <= 0.0f) {
            spdlog::warn("FixStand stand-record disabled: PARKOUR_STAND_RECORD_SECONDS <= 0");
            return;
        }

        stand_record_stop_ = false;
        stand_record_running_ = true;
        stand_record_thread_ = std::thread([this, seconds]() {
            try {
                auto record_env = make_stand_record_env();
                record_env->reset();
                const int steps = std::max(1, static_cast<int>(std::round(seconds / record_env->step_dt)));

                using clock = std::chrono::high_resolution_clock;
                const std::chrono::duration<double> desired_duration(record_env->step_dt);
                const auto dt = std::chrono::duration_cast<clock::duration>(desired_duration);
                auto sleep_till = clock::now() + dt;

                spdlog::info(
                    "FixStand stand-record started: {:.2f}s, {} steps, action output ignored.",
                    seconds,
                    steps
                );
                for (int i = 0; i < steps && !stand_record_stop_.load(); ++i) {
                    record_env->record_observation_only();
                    if ((i + 1) == 1 || (i + 1) % 50 == 0 || (i + 1) == steps) {
                        spdlog::info("FixStand stand-record {}/{}", i + 1, steps);
                    }
                    std::this_thread::sleep_until(sleep_till);
                    sleep_till += dt;
                }
                spdlog::info("FixStand stand-record complete.");
            } catch (const std::exception& e) {
                spdlog::error("FixStand stand-record failed: {}", e.what());
            }
            stand_record_running_ = false;
        });
    }

    void stop_stand_record_thread()
    {
        stand_record_stop_ = true;
        if (stand_record_thread_.joinable()) {
            stand_record_thread_.join();
        }
        stand_record_running_ = false;
    }
};

REGISTER_FSM(State_FixStand)
