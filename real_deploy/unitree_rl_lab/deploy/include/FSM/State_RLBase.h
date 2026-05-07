// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <atomic>
#include <array>
#include "FSMState.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "isaaclab/envs/mdp/terminations.h"

class State_RLBase : public FSMState
{
public:
    State_RLBase(int state_mode, std::string state_string);
    
    void enter()
    {
        // set gain
        for (int i = 0; i < env->robot->data.joint_stiffness.size(); ++i)
        {
            lowcmd->msg_.motor_cmd()[i].kp() = env->robot->data.joint_stiffness[i];
            lowcmd->msg_.motor_cmd()[i].kd() = env->robot->data.joint_damping[i];
            lowcmd->msg_.motor_cmd()[i].dq() = 0;
            lowcmd->msg_.motor_cmd()[i].tau() = 0;
        }

        env->robot->update();
        env->robot->data.parkour_command_x = 0.0f;
        policy_fault = false;
        // Start policy thread
        policy_thread_running = true;
        policy_thread = std::thread([this]{
            using clock = std::chrono::high_resolution_clock;
            const std::chrono::duration<double> desiredDuration(env->step_dt);
            const auto dt = std::chrono::duration_cast<clock::duration>(desiredDuration);

            // Initialize timing
            auto sleepTill = clock::now() + dt;
            env->reset();

            while (policy_thread_running)
            {
                try {
                    update_keyboard_velocity_command();
                    env->step();
                } catch (const std::exception& e) {
                    spdlog::critical("Policy thread fault: {}", e.what());
                    policy_fault = true;
                    policy_thread_running = false;
                    break;
                }

                // Sleep
                std::this_thread::sleep_until(sleepTill);
                sleepTill += dt;
            }
        });
    }

    void run();
    
    void exit()
    {
        policy_thread_running = false;
        if (policy_thread.joinable()) {
            policy_thread.join();
        }
    }

private:
    std::unique_ptr<isaaclab::ManagerBasedRLEnv> env;

    bool keyboard_velocity_enabled = false;
    std::array<float, 3> keyboard_velocity_command = {0.0f, 0.0f, 0.0f};
    std::array<float, 3> keyboard_velocity_step = {0.1f, 0.1f, 0.1f};
    std::array<float, 3> keyboard_velocity_limit = {0.4f, 0.3f, 0.6f};
    std::string last_keyboard_key;

    std::thread policy_thread;
    std::atomic_bool policy_thread_running = false;
    std::atomic_bool policy_fault = false;

    void update_keyboard_velocity_command();
};

REGISTER_FSM(State_RLBase)
