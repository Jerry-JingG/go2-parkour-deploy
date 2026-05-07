// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <chrono>
#include "FSMState.h"

class State_Passive : public FSMState
{
public:
    State_Passive(int state, std::string state_string = "Passive") 
    : FSMState(state, state_string) 
    {
        auto motor_mode = param::config["FSM"]["Passive"]["mode"];
        if(motor_mode.IsDefined())
        {
            auto values = motor_mode.as<std::vector<int>>();
            for(int i(0); i<values.size(); ++i)
            {
                lowcmd->msg_.motor_cmd()[i].mode() = values[i];
            }
        }

        auto cfg = param::config["FSM"][state_string];
        if (cfg["auto_transition"]) {
            const auto target_fsm = cfg["auto_transition"].as<std::string>();
            const auto after = cfg["auto_transition_after"].as<float>(0.0f);
            registered_checks.emplace_back(
                std::make_pair(
                    [this, after]()->bool {
                        const auto elapsed = std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - enter_time_
                        ).count();
                        return elapsed >= after;
                    },
                    FSMStringMap.right.at(target_fsm)
                )
            );
        }
    } 

    void enter()
    {
        enter_time_ = std::chrono::steady_clock::now();

        // set gain
        static auto kd = param::config["FSM"]["Passive"]["kd"].as<std::vector<float>>();
        for(int i(0); i < kd.size(); ++i)
        {
            auto & motor = lowcmd->msg_.motor_cmd()[i];
            motor.kp() = 0;
            motor.kd() = kd[i];
            motor.dq() = 0;
            motor.tau() = 0;
        }
    }

    void run()
    {
        for(int i(0); i < lowcmd->msg_.motor_cmd().size(); ++i)
        {
            lowcmd->msg_.motor_cmd()[i].q() = lowstate->msg_.motor_state()[i].q();
        }
    }

private:
    std::chrono::steady_clock::time_point enter_time_;
};

REGISTER_FSM(State_Passive)
