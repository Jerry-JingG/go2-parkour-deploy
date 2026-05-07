#include "FSM/State_RLBase.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"

#include <algorithm>

State_RLBase::State_RLBase(int state_mode, std::string state_string)
: FSMState(state_mode, state_string) 
{
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());
    auto deploy_cfg = YAML::LoadFile(policy_dir / "params" / "deploy.yaml");

    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        deploy_cfg,
        std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate)
    );
    if (deploy_cfg["socket_path"]) {
        env->alg = std::make_unique<isaaclab::SocketRunner>(
            deploy_cfg["socket_path"].as<std::string>(),
            deploy_cfg["socket_action_dim"].as<int>(env->action_manager->total_action_dim()),
            deploy_cfg["socket_input_name"].as<std::string>("obs")
        );
    } else {
#ifndef TXL_SOCKET_ONLY
        env->alg = std::make_unique<isaaclab::OrtRunner>(policy_dir / "exported" / "policy.onnx");
#else
        throw std::runtime_error("TXL_SOCKET_ONLY build requires socket_path in deploy.yaml");
#endif
    }

    if (deploy_cfg["keyboard_velocity"]) {
        const auto keyboard_cfg = deploy_cfg["keyboard_velocity"];
        keyboard_velocity_enabled = keyboard_cfg["enabled"].as<bool>(false) || FSMState::keyboard_control_enabled();

        auto load_vec3 = [](const YAML::Node& node, std::array<float, 3>& values) {
            if (!node) {
                return;
            }
            auto data = node.as<std::vector<float>>();
            if (data.size() != values.size()) {
                throw std::runtime_error("keyboard_velocity vector must have exactly 3 values");
            }
            std::copy(data.begin(), data.end(), values.begin());
        };

        load_vec3(keyboard_cfg["step"], keyboard_velocity_step);
        load_vec3(keyboard_cfg["limit"], keyboard_velocity_limit);
    }

    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return isaaclab::mdp::bad_orientation(env.get(), 1.0); },
            FSMStringMap.right.at("Passive")
        )
    );
    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return policy_fault.load(); },
            FSMStringMap.right.at("Passive")
        )
    );
}

void State_RLBase::update_keyboard_velocity_command()
{
    if (!keyboard_velocity_enabled) {
        return;
    }

    env->robot->data.use_velocity_command_override = true;

    if (!FSMState::keyboard) {
        env->robot->data.velocity_command_override = keyboard_velocity_command;
        return;
    }

    const auto key = FSMState::keyboard->key();
    if (key.empty()) {
        last_keyboard_key.clear();
        env->robot->data.velocity_command_override = keyboard_velocity_command;
        return;
    }

    if (key == last_keyboard_key) {
        env->robot->data.velocity_command_override = keyboard_velocity_command;
        return;
    }
    last_keyboard_key = key;

    if (key == "w" || key == "W" || key == "up") {
        keyboard_velocity_command[0] += keyboard_velocity_step[0];
    } else if (key == "s" || key == "S" || key == "down") {
        keyboard_velocity_command[0] -= keyboard_velocity_step[0];
    } else if (key == "a" || key == "A" || key == "left") {
        keyboard_velocity_command[1] += keyboard_velocity_step[1];
    } else if (key == "d" || key == "D" || key == "right") {
        keyboard_velocity_command[1] -= keyboard_velocity_step[1];
    } else if (key == "q" || key == "Q") {
        keyboard_velocity_command[2] += keyboard_velocity_step[2];
    } else if (key == "e" || key == "E") {
        keyboard_velocity_command[2] -= keyboard_velocity_step[2];
    } else if (key == " ") {
        keyboard_velocity_command = {0.0f, 0.0f, 0.0f};
    }

    for (size_t i = 0; i < keyboard_velocity_command.size(); ++i) {
        keyboard_velocity_command[i] = std::clamp(
            keyboard_velocity_command[i],
            -keyboard_velocity_limit[i],
            keyboard_velocity_limit[i]
        );
    }
    env->robot->data.velocity_command_override = keyboard_velocity_command;
    spdlog::info(
        "Keyboard velocity command: x={:.2f}, y={:.2f}, yaw={:.2f}",
        keyboard_velocity_command[0],
        keyboard_velocity_command[1],
        keyboard_velocity_command[2]
    );
}

void State_RLBase::run()
{
    auto action = env->action_manager->processed_actions();
    for(int i(0); i < env->robot->data.joint_ids_map.size(); i++) {
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
    }
}
