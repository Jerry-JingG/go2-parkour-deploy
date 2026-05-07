#include "FSM/CtrlFSM.h"
#include "FSM/State_Passive.h"
#include "FSM/State_FixStand.h"
#include "FSM/State_RLBase.h"

#include <cstdlib>

std::unique_ptr<LowCmd_t> FSMState::lowcmd = nullptr;
std::shared_ptr<LowState_t> FSMState::lowstate = nullptr;
std::shared_ptr<Keyboard> FSMState::keyboard = nullptr;

bool lowcmd_channel_has_publisher(double wait_s)
{
    auto lowcmd_sub = std::make_shared<unitree::robot::go2::subscription::LowCmd>();
    usleep(static_cast<useconds_t>(wait_s * 1e6));
    return !lowcmd_sub->isTimeout();
}

void init_fsm_state()
{
    if(lowcmd_channel_has_publisher(0.2))
    {
        spdlog::warn("The lowcmd channel is active. Trying to release Unitree motion-control service...");
        unitree::robot::go2::shutdown();
        usleep(static_cast<useconds_t>(1.0 * 1e6));
        if(lowcmd_channel_has_publisher(0.2))
        {
            spdlog::critical("The lowcmd channel is still active after ReleaseMode. Close other low-level controllers first.");
            std::exit(1);
        }
        spdlog::info("Unitree motion-control service released.");
    }
    FSMState::lowcmd = std::make_unique<LowCmd_t>();
    FSMState::lowstate = std::make_shared<LowState_t>();
    spdlog::info("Waiting for connection to robot...");
    FSMState::lowstate->wait_for_connection();
    spdlog::info("Connected to robot.");
}

int main(int argc, char** argv)
{
    // Load parameters
    auto vm = param::helper(argc, argv);

    std::cout << " --- Unitree Robotics --- \n";
    std::cout << "     Go2 Controller \n";

    // Unitree DDS Config
    unitree::robot::ChannelFactory::Instance()->Init(0, vm["network"].as<std::string>());

    init_fsm_state();
    FSMState::keyboard = std::make_shared<Keyboard>();

    // Initialize FSM
    auto fsm = std::make_unique<CtrlFSM>(param::config["FSM"]);
    fsm->start();

    std::cout << "Remote control mapping:\n";
    std::cout << "  [LT/L2 + A]  enter FixStand mode\n";
    std::cout << "  [Start]      enter parkour depth policy mode\n";
    std::cout << "  [LT/L2 + B]  return to Passive mode\n";
    std::cout << "  Left stick   forward/backward and lateral velocity\n";
    std::cout << "  Right stick  yaw velocity\n";
    std::cout << "Optional keyboard forward velocity can be enabled in policy deploy.yaml.\n";

    while (true)
    {
        sleep(1);
    }
    
    return 0;
}
