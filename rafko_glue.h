#ifndef RAFKO_GLUE_H
#define RAFKO_GLUE_H

#include "scene/main/node.h"

#include <memory>
#include <string>

#include "rafko_protocol/common.pb.h"
#include "rafko_protocol/rafko_net.pb.h"
#include "rafko_protocol/solution.pb.h"
#include "rafko_gym/services/rafko_agent.h"
#include "rafko_gym/services/rafko_net_approximizer.h"
#include "rafko_mainframe/models/rafko_service_context.h"
#include "rafko_net/services/solution_solver.h"
#include "rafko_utilities/models/data_ringbuffer.h"

#include "rafko_glue_environment.h"

class RafkoGlueEnvironment;
class RafkoGlue : public Node {
  GDCLASS(RafkoGlue, Node);

public:

  /* +++ OPTIMIZATION HOOKS +++ */
  PoolRealArray provide_current_input_values(){ /* Provides input value for the network */
    return get_script_instance()->call(StringName("provide_current_input_values"));
  }
  void apply_network_output(PoolRealArray net_output){ /* Apply network output to the environment */
    get_script_instance()->call(StringName("apply_network_output"), net_output);
  }
  void push_state(){
    get_script_instance()->call(StringName("push_state"));
  }
  void pop_state(){
    get_script_instance()->call(StringName("pop_state"));
  }
  double get_current_fitness(){
    return get_script_instance()->call(StringName("get_current_fitness"));
  }
  void set_starting_state();
  void set_evaluation_parameters(uint32 full_evaluation_loops, uint32 stochastic_evaluation_loops);
  /* --- OPTIMIZATION HOOKS --- */

  /* +++ NETWORK_HANDLING +++ */
  bool create_network(int input_size, PoolVector<int> layer_numbers, double expected_input_range);
  /* TODO#2: include PoolVector<rafko_net::Transfer_functions> layer_functions */
  /* TODO#1: Load and save network */
  void optimize_step(){
    if(ready()){
      if(optimizer){
        optimizer->collect_approximates_from_weight_gradients();
        optimizer->apply_fragment();
        solver_deprecated = true;
      }else store_error("Trying to optimize a non-existing network!");
    }else{
      std::string s = (
        "Trying to start optimization before the pre-requisites are met: "
        +(start_state_set)?"":"Start state not set;"
        +(eval_params_set)?"":"evaluation paramters not set nost set;"
      );
      store_error(s);
    }
  }
  PoolRealArray calculate(PoolRealArray network_input, bool reset);
  /* --- NETWORK_HANDLING --- */

  StringName get_latest_error_message(){
    return StringName(latest_error_message.c_str());
  }
  RafkoGlue(){
    /* TODO#3: Use arenas: (void)context.set_arena_ptr(&arena); */
    environment = std::make_unique<RafkoGlueEnvironment>(*this);
  }
  ~RafkoGlue(){
    environment.reset();
    if(network)network.reset();
    if(optimizer)optimizer.reset();
    if(solver)solver.reset();
  }

  static PoolRealArray toPoolArray(const std::vector<double> vec);
  static std::vector<double> toStdVec(const PoolRealArray arr);

protected:
  static void _bind_methods();
private:
  /* TODO#3: Use arenas: google::protobuf::Arena arena; */
  rafko_mainframe::RafkoServiceContext context;
  std::unique_ptr<RafkoGlueEnvironment> environment;
  std::unique_ptr<rafko_net::RafkoNet> network;
  std::unique_ptr<rafko_gym::RafkoNetApproximizer> optimizer;
  std::unique_ptr<rafko_net::Solution> solution;
  std::unique_ptr<rafko_net::SolutionSolver> solver;
  bool solver_deprecated = true;
  std::string latest_error_message;
  int error_count;
  bool start_state_set = false;
  bool eval_params_set = false;

  void refresh_solver();
  void store_error(std::string err){
    ++error_count;
    latest_error_message = "error[" + std::string(std::to_string(error_count)) + "]:" + err;
  }

  bool ready(){ /* Returns with true if everything is set to optimize! */
    return(eval_params_set && start_state_set && network && optimizer);
  }
};

#endif /* RAFKO_GLUE_H */
