#include "rafko_glue.h"

#include <vector>
#include <cmath>
#include <thread>

#include "rafko_net/services/rafko_net_builder.h"
#include "rafko_net/services/solution_builder.h"
#include "rafko_utilities/models/const_vector_subrange.h"

void RafkoGlue::_bind_methods() {
  ClassDB::bind_method(D_METHOD("get_latest_error_message"), &RafkoGlue::get_latest_error_message);
  ClassDB::bind_method(D_METHOD("create_network"), &RafkoGlue::create_network);
  ClassDB::bind_method(D_METHOD("provide_current_input_values"), &RafkoGlue::provide_current_input_values);
  ClassDB::bind_method(D_METHOD("apply_network_output"), &RafkoGlue::apply_network_output);
  ClassDB::bind_method(D_METHOD("calculate"), &RafkoGlue::calculate);
  ClassDB::bind_method(D_METHOD("start_optimization"), &RafkoGlue::start_optimization);
  ClassDB::bind_method(D_METHOD("stop_optimization"), &RafkoGlue::stop_optimization);
  ClassDB::bind_method(D_METHOD("get_current_fitness"), &RafkoGlue::get_current_fitness);
  ClassDB::bind_method(D_METHOD("set_evaluation_parameters"), &RafkoGlue::set_evaluation_parameters);
  ClassDB::bind_method(D_METHOD("notify_actions_processed"), &RafkoGlue::notify_actions_processed);
  ClassDB::bind_method(D_METHOD("notify_pop_processed"), &RafkoGlue::notify_pop_processed);
  ClassDB::bind_method(D_METHOD("push_state"), &RafkoGlue::push_state);
  ClassDB::bind_method(D_METHOD("pop_state"), &RafkoGlue::pop_state);
}

PoolRealArray RafkoGlue::provide_current_input_values(){ /* Provides input value for the network */
  return get_script_instance()->call(StringName("provide_current_input_values"));
}

void RafkoGlue::apply_network_output(PoolRealArray net_output){ /* Apply network output to the environment */
  get_script_instance()->call(StringName("apply_network_output"), net_output);
}

void RafkoGlue::push_state(){
  get_script_instance()->call(StringName("push_state"));
}

void RafkoGlue::pop_state(){
  get_script_instance()->call(StringName("pop_state"));
}

double RafkoGlue::get_current_fitness(){
  return get_script_instance()->call(StringName("get_current_fitness"));
}

void RafkoGlue::set_evaluation_parameters(uint32 full_evaluation_loops, uint32 stochastic_evaluation_loops){
  environment->set_evaluation_parameters(full_evaluation_loops, stochastic_evaluation_loops);
  eval_params_set = true;
}

void RafkoGlue::notify_actions_processed(){
  environment->notify_actions_processed();
}

void RafkoGlue::notify_pop_processed(){
  environment->notify_pop_processed();
}

void RafkoGlue::stop_optimization(){
  std::lock_guard<std::mutex> my_lock(optimization_control_mutex);
  std::cout << "Stop opt requested!" << std::endl;
  if(do_optimization){
    do_optimization = false;
    synchroniser.notify_one();
  }
}

void RafkoGlue::start_optimization(){
  if(ready()){
    if(optimizer){
      {
        std::lock_guard<std::mutex> my_lock(optimization_control_mutex);
        std::cout << "Start opt requested!" << std::endl;
        do_optimization = true;
        synchroniser.notify_one();
      }
    }else store_error("Trying to optimize a non-existing network!");
  }else{
    std::string s = (
      "Trying to start optimization before the pre-requisites are met: "
      +(eval_params_set)?"":"evaluation paramters not set nost set;"
    );
    store_error(s);
  }
}

void RafkoGlue::optimize_thread_function(){
  bool local_do_optimization = false;
  bool local_continue_running = true;
  while(local_continue_running)
  {
    if(local_do_optimization){
      {
        std::lock_guard<std::mutex> my_lock(optimization_control_mutex);
        std::cout << "[g]-->\tAn iteration started!" << std::endl;
        optimization_in_progress = true;
      }

      {
        std::lock_guard<std::mutex> my_lock(network_mutex);
        optimizer->collect_approximates_from_weight_gradients();
        optimizer->apply_fragment();
      }
      {
        std::lock_guard<std::mutex> my_lock(optimization_control_mutex);
        std::cout << "[g]-->\tAn iteration finished!" << std::endl;
        solver_deprecated = true;
        if(!local_do_optimization)optimization_in_progress = false;
      }
    }/*if(do_optimization)*/
    {
      std::unique_lock<std::mutex> my_lock(optimization_control_mutex);
      local_do_optimization = do_optimization;
      local_continue_running = continue_running;
      if((!local_do_optimization)&&(local_continue_running)){
        synchroniser.wait(my_lock, [&](){
          local_do_optimization = do_optimization;
          local_continue_running = continue_running;
          return (do_optimization || (!continue_running));
        });
      }
    }
  }/* while(continue_running) */
}

void RafkoGlue::refresh_solver(){
  if(network){
    if(solver)solver.reset();
    if(solution)solution.reset();

    {
      std::lock_guard<std::mutex> my_lock(network_mutex);
      solution = std::unique_ptr<rafko_net::Solution>(
        rafko_net::SolutionBuilder(context).build(*network)
      );
    }

    solver = std::unique_ptr<rafko_net::SolutionSolver>(
      rafko_net::SolutionSolver::Builder(*solution, context).build()
    );

    {
      std::lock_guard<std::mutex> my_lock(optimization_control_mutex);
      solver_deprecated = false;
    }
  }else store_error("Trying to build a Solution for a non-existing network!");
}

bool RafkoGlue::create_network(int input_size, PoolVector<int> layer_numbers, double expected_input_range) {
  stop_optimization();

  /* Build Transfer_functions and layer sizes layers */
  std::vector<unsigned int> layer_structure(layer_numbers.size());
  std::vector<std::vector<rafko_net::Transfer_functions>> layer_transfers(
    layer_numbers.size(), {rafko_net::transfer_function_selu}
  );

  for(int i=0; i<layer_numbers.size();++i){
    layer_structure[i] = layer_numbers[i];
  }

  if(network){ /* Already have a network! */
    {
      std::unique_lock<std::mutex> my_lock(optimization_control_mutex);
      do_optimization = false;
    }
    bool local_optimization_in_progress = true;
    while(local_optimization_in_progress){ /*TODO#4:Eliminate busy waiting! */
      {
        std::lock_guard<std::mutex> my_lock(optimization_control_mutex);
        local_optimization_in_progress = optimization_in_progress;
      }
    }
    network.reset();
  }
  network = std::unique_ptr<rafko_net::RafkoNet>(
    rafko_net::RafkoNetBuilder(context)
    .input_size(input_size)
    .expected_input_range(std::abs(expected_input_range))
    .allowed_transfer_functions_by_layer(layer_transfers)
    .dense_layers(layer_structure)
  );

  if(optimizer)optimizer.reset();
  optimizer = std::make_unique<rafko_gym::RafkoNetApproximizer>(
    context, *network, *environment, rafko_net::weight_updater_amsgrad
  );

  return true;
}

PoolRealArray RafkoGlue::calculate(PoolRealArray network_input, bool reset){
  if(network){
    if(network_input.size() < static_cast<sint32>(network->input_data_size())){
      latest_error_message = "Not enough input provided for network!";
      return PoolRealArray();
    }
    if(!network) return PoolRealArray();
    bool local_solver_deprecated;
    {
      std::lock_guard<std::mutex> my_lock(optimization_control_mutex);
      local_solver_deprecated = solver_deprecated;
    }
    if(local_solver_deprecated)refresh_solver();
    std::vector<double> inputs = toStdVec(network_input);
    rafko_utilities::ConstVectorSubrange<> result = solver->solve(
      {inputs.begin(), inputs.begin() + network->input_data_size()}, reset
    );
    return toPoolArray({ result.begin(), result.end() });
  }else store_error("Trying to calculate a non-existing network!");

  return PoolRealArray();
}

RafkoGlue::~RafkoGlue(){
  stop_optimization();

  {
    std::lock_guard<std::mutex> my_lock(optimization_control_mutex);
    continue_running = false;
    synchroniser.notify_one();
  }
  optimize_thread.join();

  environment.reset();
  if(network)network.reset(); /* At this point optimization is stopped so no need to use net mutex */
  if(optimizer)optimizer.reset();
  if(solver)solver.reset();
}

PoolRealArray RafkoGlue::toPoolArray(std::vector<double> vec){
  PoolRealArray result;
  for(double& element : vec)result.push_back(element);
  return result;
}

std::vector<double> RafkoGlue::toStdVec(PoolRealArray arr){
  std::vector<double> result;
  for(int i = 0; i < arr.size(); ++i){
    result.push_back(arr[i]);
  }
  return result;
}
