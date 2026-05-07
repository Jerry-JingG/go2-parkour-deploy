#include <librealsense2/rs.hpp>

#include <arpa/inet.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <chrono>
#include <csignal>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr uint32_t kMagic = 0x48545044; // "DPTH" little-endian
volatile std::sig_atomic_t g_running = 1;

struct Options {
    std::string socket_path = "/tmp/go2_realsense_depth.sock";
    std::string serial;
    int width = 424;
    int height = 240;
    int fps = 30;
    int timeout_ms = 200;
};

void on_signal(int) {
    g_running = 0;
}

Options parse_args(int argc, char** argv) {
    Options opts;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const std::string& name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for " + name);
            }
            return argv[++i];
        };

        if (arg == "--socket_path") {
            opts.socket_path = require_value(arg);
        } else if (arg == "--serial") {
            opts.serial = require_value(arg);
        } else if (arg == "--width") {
            opts.width = std::stoi(require_value(arg));
        } else if (arg == "--height") {
            opts.height = std::stoi(require_value(arg));
        } else if (arg == "--fps") {
            opts.fps = std::stoi(require_value(arg));
        } else if (arg == "--timeout_ms") {
            opts.timeout_ms = std::stoi(require_value(arg));
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: realsense_depth_streamer [options]\n"
                << "  --socket_path PATH   Unix socket path [/tmp/go2_realsense_depth.sock]\n"
                << "  --serial SERIAL      Optional RealSense serial\n"
                << "  --width WIDTH        Depth width [424]\n"
                << "  --height HEIGHT      Depth height [240]\n"
                << "  --fps FPS            Depth FPS [30]\n"
                << "  --timeout_ms MS      Frame timeout [200]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }
    return opts;
}

void send_all(int fd, const void* data, size_t size) {
    const char* ptr = static_cast<const char*>(data);
    size_t sent = 0;
    while (sent < size) {
        const ssize_t n = ::send(fd, ptr + sent, size - sent, MSG_NOSIGNAL);
        if (n <= 0) {
            throw std::runtime_error("socket send failed");
        }
        sent += static_cast<size_t>(n);
    }
}

int make_server_socket(const std::string& path) {
    if (path.size() >= sizeof(sockaddr_un::sun_path)) {
        throw std::runtime_error("Socket path is too long: " + path);
    }

    const int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) {
        throw std::runtime_error("failed to create socket");
    }

    ::unlink(path.c_str());

    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    std::strncpy(addr.sun_path, path.c_str(), sizeof(addr.sun_path) - 1);

    if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        ::close(fd);
        throw std::runtime_error("failed to bind socket: " + path);
    }
    if (::listen(fd, 1) != 0) {
        ::close(fd);
        throw std::runtime_error("failed to listen on socket: " + path);
    }
    return fd;
}

void stream_frames(int client_fd, rs2::pipeline& pipe, float depth_scale, int timeout_ms) {
    while (g_running) {
        rs2::frameset frames;
        try {
            frames = pipe.wait_for_frames(static_cast<unsigned int>(timeout_ms));
        } catch (const rs2::error&) {
            continue;
        }

        rs2::depth_frame depth = frames.get_depth_frame();
        if (!depth) {
            continue;
        }

        const int width = depth.get_width();
        const int height = depth.get_height();
        const auto* raw = reinterpret_cast<const uint16_t*>(depth.get_data());
        std::vector<float> depth_m(static_cast<size_t>(width) * static_cast<size_t>(height));
        for (size_t i = 0; i < depth_m.size(); ++i) {
            depth_m[i] = static_cast<float>(raw[i]) * depth_scale;
        }

        const uint32_t header[3] = {
            kMagic,
            static_cast<uint32_t>(width),
            static_cast<uint32_t>(height),
        };
        send_all(client_fd, header, sizeof(header));
        send_all(client_fd, depth_m.data(), depth_m.size() * sizeof(float));
    }
}

} // namespace

int main(int argc, char** argv) {
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);

    try {
        const Options opts = parse_args(argc, argv);

        rs2::config cfg;
        if (!opts.serial.empty()) {
            cfg.enable_device(opts.serial);
        }
        cfg.enable_stream(RS2_STREAM_DEPTH, opts.width, opts.height, RS2_FORMAT_Z16, opts.fps);

        rs2::pipeline pipe;
        rs2::pipeline_profile profile = pipe.start(cfg);
        auto depth_sensor = profile.get_device().first<rs2::depth_sensor>();
        const float depth_scale = depth_sensor.get_depth_scale();

        const int server_fd = make_server_socket(opts.socket_path);
        std::cout << "RealSense depth streamer ready on " << opts.socket_path << "\n"
                  << "stream=" << opts.width << "x" << opts.height << "@" << opts.fps
                  << " depth_scale=" << depth_scale << std::endl;

        while (g_running) {
            const int client_fd = ::accept(server_fd, nullptr, nullptr);
            if (client_fd < 0) {
                if (g_running) {
                    std::cerr << "accept failed\n";
                }
                continue;
            }
            std::cout << "depth client connected" << std::endl;
            try {
                stream_frames(client_fd, pipe, depth_scale, opts.timeout_ms);
            } catch (const std::exception& exc) {
                std::cerr << "depth client disconnected: " << exc.what() << std::endl;
            }
            ::close(client_fd);
        }

        ::close(server_fd);
        ::unlink(opts.socket_path.c_str());
        pipe.stop();
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "ERROR: " << exc.what() << std::endl;
        return 1;
    }
}
