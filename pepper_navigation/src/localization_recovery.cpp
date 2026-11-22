/* localization_recovery.cpp
 *
 * One "I am lost, fix it" service that works whichever localization backend
 * is running.
 *
 * Exposes /localization_recover (std_srvs/Trigger) in front of all of them and
 * dispatches to whatever is actually up, so an operator switching between
 * Nav2 profiles does not have to remember one recovery procedure per backend:
 *
 *   fastloc   /relocalize (std_srvs/Trigger), re-arms the ScanContext search
 *   pointloc  /relocalize as well; the Point-LIO localizer is a port of the
 *             FAST-LIO one and exposes the same service
 *   amcl      /reinitialize_global_localization (std_srvs/Empty), scatters
 *             particles for a global re-draw
 *
 * Parameters (config/pepper_navigation_configuration.yaml):
 *   backend          fastloc | pointloc | amcl, or auto to detect
 *                    it from the graph; each Nav2 profile pins its own
 *   service_timeout  s to wait for the backend's service, and for its answer
 *
 * Author: Yohannes Tadesse Haile
 * Affiliation: Carnegie Mellon University Africa
 * Email: yohatad123@gmail.com
 * Date: November 17, 2026
 * Version: v1.0
 *
 * Copyright (C) 2025 Carnegie Mellon University Africa
 * This software is provided 'as-is' for research and educational purposes
 * within the DEC project.
 */

#include <chrono>
#include <future>
#include <memory>
#include <set>
#include <string>
#include <utility>

#include <rclcpp/rclcpp.hpp>
#include <std_srvs/srv/empty.hpp>
#include <std_srvs/srv/trigger.hpp>

namespace
{
constexpr char kFastlocService[] = "/relocalize";
constexpr char kAmclService[] = "/reinitialize_global_localization";
}  // namespace

class LocalizationRecovery : public rclcpp::Node
{
public:
  LocalizationRecovery()
  : Node("localization_recovery")
  {
    // 'auto' resolves from the graph; each launch file pins it.
    backend_ = declare_parameter<std::string>("backend", "auto");
    timeout_ = std::chrono::duration<double>(declare_parameter<double>("service_timeout", 5.0));

    // The service callback blocks on a client call, so the two must run
    // concurrently: in one callback group the answer could never be taken.
    srv_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    cli_group_ = create_callback_group(rclcpp::CallbackGroupType::Reentrant);

    fastloc_client_ = create_client<std_srvs::srv::Trigger>(
      kFastlocService, rmw_qos_profile_services_default, cli_group_);
    amcl_client_ = create_client<std_srvs::srv::Empty>(
      kAmclService, rmw_qos_profile_services_default, cli_group_);

    srv_ = create_service<std_srvs::srv::Trigger>(
      "/localization_recover",
      [this](
        const std_srvs::srv::Trigger::Request::SharedPtr,
        std_srvs::srv::Trigger::Response::SharedPtr res) {recover(*res);},
      rmw_qos_profile_services_default, srv_group_);

    RCLCPP_INFO(get_logger(), "/localization_recover ready (backend=%s)", backend_.c_str());
  }

private:
  // Which localization stack is live, by what other nodes actually SERVE.
  // Not the graph's plain service list: that also holds services this node
  // only has a CLIENT for, so it listed our own /relocalize client and every
  // auto-detect resolved to fastloc with nothing running at all. Asking per
  // node returns servers only.
  std::string detectBackend()
  {
    const auto graph = get_node_graph_interface();
    std::set<std::string> served;
    std::set<std::string> live;
    for (const auto & [name, ns] : graph->get_node_names_and_namespaces()) {
      live.insert(name);
      if (name == get_name() && ns == get_namespace()) {
        continue;
      }
      try {
        for (const auto & srv : get_service_names_and_types_by_node(name, ns)) {
          served.insert(srv.first);
        }
      } catch (const std::exception &) {
        // The node vanished between listing and querying: normal churn.
      }
    }
    if (served.count(kFastlocService)) {
      // fastloc and pointloc both serve /relocalize, so the service alone
      // cannot tell them apart; who is running can.
      return live.count("point_lio_localization") ? "pointloc" : "fastloc";
    }
    if (served.count(kAmclService)) {
      return "amcl";
    }
    return "";
  }

  // Calls a service, waiting at most service_timeout for it to appear and
  // for its answer. Empty message on success, else what went wrong.
  template<typename ServiceT>
  std::string call(
    const typename rclcpp::Client<ServiceT>::SharedPtr & client, const char * service,
    typename ServiceT::Response::SharedPtr & response)
  {
    if (!client->wait_for_service(timeout_)) {
      return std::string(service) + " did not appear";
    }
    auto future = client->async_send_request(std::make_shared<typename ServiceT::Request>());
    if (future.wait_for(timeout_) != std::future_status::ready) {
      client->remove_pending_request(future);
      return std::string(service) + " timed out";
    }
    response = future.get();
    return "";
  }

  void recover(std_srvs::srv::Trigger::Response & res)
  {
    const std::string backend = backend_ != "auto" ? backend_ : detectBackend();

    if (backend == "fastloc" || backend == "pointloc") {
      std_srvs::srv::Trigger::Response::SharedPtr result;
      const auto error = call<std_srvs::srv::Trigger>(fastloc_client_, kFastlocService, result);
      if (!error.empty()) {
        res.success = false;
        res.message = error;
      } else {
        const char * node = backend == "pointloc" ? "pointlio_localization" :
          "fastlio_localization";
        res.success = result->success;
        res.message = std::string(node) + ": " + result->message;
      }
    } else if (backend == "amcl") {
      std_srvs::srv::Empty::Response::SharedPtr result;
      const auto error = call<std_srvs::srv::Empty>(amcl_client_, kAmclService, result);
      res.success = error.empty();
      res.message = error.empty() ?
        "amcl: particles re-scattered globally. Drive the robot so the filter can converge." :
        error;
    } else {
      res.success = false;
      res.message =
        "no known localization backend detected (looked for /relocalize, "
        "/reinitialize_global_localization)";
    }

    if (res.success) {
      RCLCPP_INFO(get_logger(), "/localization_recover -> %s", res.message.c_str());
    } else {
      RCLCPP_WARN(get_logger(), "/localization_recover -> %s", res.message.c_str());
    }
  }

  std::string backend_;
  std::chrono::duration<double> timeout_;
  rclcpp::CallbackGroup::SharedPtr srv_group_;
  rclcpp::CallbackGroup::SharedPtr cli_group_;
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr fastloc_client_;
  rclcpp::Client<std_srvs::srv::Empty>::SharedPtr amcl_client_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr srv_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::executors::MultiThreadedExecutor executor;
  auto node = std::make_shared<LocalizationRecovery>();
  executor.add_node(node);
  executor.spin();
  rclcpp::shutdown();
  return 0;
}
