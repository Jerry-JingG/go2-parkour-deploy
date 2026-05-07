#pragma once

#include "Types.h"
#include "param.h"
#include "FSM/BaseState.h"
#include "isaaclab/devices/keyboard/keyboard.h"
#include "unitree_joystick_dsl.hpp"

#include <cctype>
#include <cstdlib>
#include <string>
#include <vector>

class FSMState : public BaseState
{
public:
    static std::string normalize_keyboard_key(std::string key)
    {
        if (key == " ") {
            return "space";
        }
        for (char& c : key) {
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
        return key;
    }

    static bool keyboard_control_enabled()
    {
        const char* value = std::getenv("UNITREE_RL_LAB_KEYBOARD_CONTROL");
        if (!value) {
            return false;
        }
        const std::string enabled = normalize_keyboard_key(value);
        return enabled == "1" || enabled == "true" || enabled == "yes" || enabled == "on";
    }

    static bool keyboard_on_pressed(const std::vector<std::string>& keys)
    {
        if (!keyboard_control_enabled() || !keyboard || !keyboard->on_pressed) {
            return false;
        }

        const std::string pressed_key = normalize_keyboard_key(keyboard->key());
        for (const auto& key : keys) {
            if (pressed_key == normalize_keyboard_key(key)) {
                return true;
            }
        }
        return false;
    }

    FSMState(int state, std::string state_string) 
    : BaseState(state, state_string) 
    {
        spdlog::info("Initializing State_{} ...", state_string);

        auto transitions = param::config["FSM"][state_string]["transitions"];

        if(transitions)
        {
            auto transition_map = transitions.as<std::map<std::string, std::string>>();

            for(auto it = transition_map.begin(); it != transition_map.end(); ++it)
            {
                std::string target_fsm = it->first;
                if(!FSMStringMap.right.count(target_fsm))
                {
                    spdlog::warn("FSM State_'{}' not found in FSMStringMap!", target_fsm);
                    continue;
                }

                int fsm_id = FSMStringMap.right.at(target_fsm);

                std::string condition = it->second;
                unitree::common::dsl::Parser p(condition);
                auto ast = p.Parse();
                auto func = unitree::common::dsl::Compile(*ast);
                registered_checks.emplace_back(
                    std::make_pair(
                        [func]()->bool{ return func(FSMState::lowstate->joystick); },
                        fsm_id
                    )
                );
            }
        }

        auto keyboard_transitions = param::config["FSM"][state_string]["keyboard_transitions"];
        if (keyboard_transitions && keyboard_control_enabled())
        {
            for (auto it = keyboard_transitions.begin(); it != keyboard_transitions.end(); ++it)
            {
                std::string target_fsm = it->first.as<std::string>();
                if(!FSMStringMap.right.count(target_fsm))
                {
                    spdlog::warn("FSM State_'{}' not found in FSMStringMap!", target_fsm);
                    continue;
                }

                std::vector<std::string> keys;
                if (it->second.IsSequence()) {
                    for (auto key : it->second) {
                        keys.push_back(key.as<std::string>());
                    }
                } else {
                    keys.push_back(it->second.as<std::string>());
                }

                int fsm_id = FSMStringMap.right.at(target_fsm);
                registered_checks.emplace_back(
                    std::make_pair(
                        [keys]()->bool{ return FSMState::keyboard_on_pressed(keys); },
                        fsm_id
                    )
                );
            }
        }

        // register for all states
        registered_checks.emplace_back(
            std::make_pair(
                []()->bool{ return lowstate->isTimeout(); },
                FSMStringMap.right.at("Passive")
            )
        );
    }

    void pre_run()
    {
        lowstate->update();
        if(keyboard) keyboard->update();
    }

    void post_run()
    {
        lowcmd->unlockAndPublish();
    }

    static std::unique_ptr<LowCmd_t> lowcmd;
    static std::shared_ptr<LowState_t> lowstate;
    static std::shared_ptr<Keyboard> keyboard;
};
