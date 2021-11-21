#include "rafko_glue.h"

#include <vector>
#include <cmath>

#include "rafko_net/services/rafko_net_builder.h"
#include "rafko_net/services/solution_builder.h"


void RafkoGlue::_bind_methods() {
  ClassDB::bind_method(D_METHOD("optimize_step"), &RafkoGlue::optimize_step);
  ClassDB::bind_method(D_METHOD("create_network"), &RafkoGlue::create_network);
  ClassDB::bind_method(D_METHOD("push_state"), &RafkoGlue::push_state);
  ClassDB::bind_method(D_METHOD("pop_state"), &RafkoGlue::pop_state);
  ClassDB::bind_method(D_METHOD("provide_current_input_values"), &RafkoGlue::provide_current_input_values);
  ClassDB::bind_method(D_METHOD("apply_network_output"), &RafkoGlue::apply_network_output);
}

void RafkoGlue::refresh_solver(){
  if(network){
    if(solver)solver.reset();
    if(solution)solution.reset();

    solution = std::unique_ptr<rafko_net::Solution>(
      rafko_net::SolutionBuilder(context).build(*network)
    );
    solver = std::unique_ptr<rafko_net::SolutionSolver>(
      rafko_net::SolutionSolver::Builder(*solution, context).build()
    );
    solver_deprecated = false;
  }
}

bool RafkoGlue::create_network(int input_size, PoolVector<int> layer_numbers, double expected_input_range) {
  if(network)network.reset();

  /* Build Transfer_functions and layer sizes layers */
  std::vector<unsigned int> layer_structure(layer_numbers.size());
  std::vector<std::vector<rafko_net::Transfer_functions>> layer_transfers(
    layer_numbers.size(), {rafko_net::transfer_function_selu}
  );
  for(int i=0; i<layer_numbers.size();++i){
    layer_structure[i] = layer_numbers[i];
  }

  network = std::unique_ptr<rafko_net::RafkoNet>(
    rafko_net::RafkoNetBuilder(context)
    .input_size(input_size)
    .expected_input_range(std::abs(expected_input_range))
    .allowed_transfer_functions_by_layer(layer_transfers)
    .dense_layers(layer_structure)
  );

  optimizer = std::make_unique<rafko_gym::RafkoNetApproximizer>(
    context, *network, *environment, rafko_net::weight_updater_amsgrad
  );
  return true;
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
