// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#ifndef TXL_SOCKET_ONLY
#include "onnxruntime_cxx_api.h"
#endif
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <sys/un.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <vector>

namespace isaaclab
{

class Algorithms
{
public:
    virtual std::vector<float> act(std::unordered_map<std::string, std::vector<float>> obs) = 0;

    std::vector<float> get_action()
    {
        std::lock_guard<std::mutex> lock(act_mtx_);
        return action;
    }
    
    std::vector<float> action;
protected:
    std::mutex act_mtx_;
};

#ifndef TXL_SOCKET_ONLY
class OrtRunner : public Algorithms
{
public:
    OrtRunner(std::string model_path)
    {
        // Init Model
        env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "onnx_model");
        session_options.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);

        session = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);

        for (size_t i = 0; i < session->GetInputCount(); ++i) {
            Ort::TypeInfo input_type = session->GetInputTypeInfo(i);
            input_shapes.push_back(input_type.GetTensorTypeAndShapeInfo().GetShape());
            auto input_name = session->GetInputNameAllocated(i, allocator);
            input_names.push_back(input_name.release());
        }

        for (const auto& shape : input_shapes) {
            size_t size = 1;
            for (const auto& dim : shape) {
                size *= dim;
            }
            input_sizes.push_back(size);
        }

        // Get output shape
        Ort::TypeInfo output_type = session->GetOutputTypeInfo(0);
        output_shape = output_type.GetTensorTypeAndShapeInfo().GetShape();
        auto output_name = session->GetOutputNameAllocated(0, allocator);
        output_names.push_back(output_name.release());

        action.resize(output_shape[1]);
    }

    std::vector<float> act(std::unordered_map<std::string, std::vector<float>> obs)
    {
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

        // make sure all input names are in obs
        for (const auto& name : input_names) {
            if (obs.find(name) == obs.end()) {
                throw std::runtime_error("Input name " + std::string(name) + " not found in observations.");
            }
        }

        // Create input tensors
        std::vector<Ort::Value> input_tensors;
        for(int i(0); i<input_names.size(); ++i)
        {
            const std::string name_str(input_names[i]);
            auto& input_data = obs.at(name_str);
            auto input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_data.data(), input_sizes[i], input_shapes[i].data(), input_shapes[i].size());
            input_tensors.push_back(std::move(input_tensor));
        }

        // Run the model
        auto output_tensor = session->Run(Ort::RunOptions{nullptr}, input_names.data(), input_tensors.data(), input_tensors.size(), output_names.data(), 1);

        // Copy output data
        auto floatarr = output_tensor.front().GetTensorMutableData<float>();
        std::lock_guard<std::mutex> lock(act_mtx_);
        std::memcpy(action.data(), floatarr, output_shape[1] * sizeof(float));
        return action;
    }

private:
    Ort::Env env;
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;
    Ort::AllocatorWithDefaultOptions allocator;

    std::vector<const char*> input_names;
    std::vector<const char*> output_names;

    std::vector<std::vector<int64_t>> input_shapes;
    std::vector<int64_t> input_sizes;
    std::vector<int64_t> output_shape;
};
#endif

class SocketRunner : public Algorithms
{
public:
    SocketRunner(std::string socket_path, int action_dim = 12, std::string input_name = "obs")
    : socket_path_(std::move(socket_path)), input_name_(std::move(input_name)), action_dim_(action_dim)
    {
        action.resize(action_dim_, 0.0f);
    }

    ~SocketRunner()
    {
        if (fd_ >= 0) {
            ::close(fd_);
        }
    }

    std::vector<float> act(std::unordered_map<std::string, std::vector<float>> obs)
    {
        const auto& input = select_input(obs);
        std::vector<float> output;
        try {
            output = request_once(input);
        } catch (const std::exception&) {
            connect_socket();
            output = request_once(input);
        }

        if (static_cast<int>(output.size()) != action_dim_) {
            throw std::runtime_error(
                "SocketRunner expected action_dim=" + std::to_string(action_dim_) +
                ", got " + std::to_string(output.size())
            );
        }

        std::lock_guard<std::mutex> lock(act_mtx_);
        action = output;
        return action;
    }

private:
    const std::vector<float>& select_input(const std::unordered_map<std::string, std::vector<float>>& obs) const
    {
        auto it = obs.find(input_name_);
        if (it != obs.end()) {
            return it->second;
        }
        if (obs.size() == 1) {
            return obs.begin()->second;
        }
        throw std::runtime_error("SocketRunner input '" + input_name_ + "' not found in observations.");
    }

    void connect_socket()
    {
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }

        sockaddr_un addr{};
        if (socket_path_.size() >= sizeof(addr.sun_path)) {
            throw std::runtime_error("Socket path is too long: " + socket_path_);
        }

        fd_ = ::socket(AF_UNIX, SOCK_STREAM, 0);
        if (fd_ < 0) {
            throw std::runtime_error("Failed to create Unix socket: " + std::string(std::strerror(errno)));
        }

        addr.sun_family = AF_UNIX;
        std::strncpy(addr.sun_path, socket_path_.c_str(), sizeof(addr.sun_path) - 1);

        int last_errno = 0;
        for (int i = 0; i < 300; ++i) {
            if (::connect(fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0) {
                return;
            }
            last_errno = errno;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        ::close(fd_);
        fd_ = -1;
        throw std::runtime_error(
            "Failed to connect to " + socket_path_ + ": " + std::string(std::strerror(last_errno))
        );
    }

    void send_all(const void* data, size_t size)
    {
        const char* ptr = static_cast<const char*>(data);
        size_t sent = 0;
        while (sent < size) {
            const ssize_t n = ::send(fd_, ptr + sent, size - sent, MSG_NOSIGNAL);
            if (n <= 0) {
                throw std::runtime_error("Socket send failed: " + std::string(std::strerror(errno)));
            }
            sent += static_cast<size_t>(n);
        }
    }

    void recv_all(void* data, size_t size)
    {
        char* ptr = static_cast<char*>(data);
        size_t received = 0;
        while (received < size) {
            const ssize_t n = ::recv(fd_, ptr + received, size - received, 0);
            if (n <= 0) {
                throw std::runtime_error("Socket recv failed: " + std::string(n == 0 ? "peer closed" : std::strerror(errno)));
            }
            received += static_cast<size_t>(n);
        }
    }

    std::vector<float> request_once(const std::vector<float>& input)
    {
        if (fd_ < 0) {
            connect_socket();
        }
        const uint32_t n_in = static_cast<uint32_t>(input.size());
        send_all(&n_in, sizeof(n_in));
        if (!input.empty()) {
            send_all(input.data(), input.size() * sizeof(float));
        }

        uint32_t n_out = 0;
        recv_all(&n_out, sizeof(n_out));
        std::vector<float> output(n_out);
        if (!output.empty()) {
            recv_all(output.data(), output.size() * sizeof(float));
        }
        return output;
    }

    std::string socket_path_;
    std::string input_name_;
    int action_dim_;
    int fd_ = -1;
};
};
