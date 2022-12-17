#include "rafko_glue.h"

#include <vector>
#include <cmath>
#include <optional>

#include "rafko_net/services/rafko_net_builder.hpp"
#include "rafko_net/services/solution_builder.hpp"
#include "rafko_utilities/models/const_vector_subrange.hpp"

namespace {
  PackedFloat32Array toPoolArray(const std::vector<double>& vec){
    PackedFloat32Array result;
    for(const double& element : vec)result.push_back(element);
    return result;
  }

  std::vector<double> toStdVec(const PackedFloat32Array& arr){
    std::vector<double> result;
    for(int i = 0; i < arr.size(); ++i){
      result.push_back(arr[i]);
    }
    return result;
  }
} /* namespace */

void RafkoGlue::_bind_methods() {
  ClassDB::bind_method(D_METHOD("configure_network"), &RafkoGlue::configure_network);
  ClassDB::bind_method(D_METHOD("configure_environment"), &RafkoGlue::configure_environment);
  ClassDB::bind_method(D_METHOD("configure_trainer"), &RafkoGlue::configure_trainer);
  ClassDB::bind_method(D_METHOD("calculate"), &RafkoGlue::calculate);
  ClassDB::bind_method(D_METHOD("iterate"), &RafkoGlue::iterate);
  ClassDB::bind_method(D_METHOD("reset_environment"), &RafkoGlue::reset_environment);
  ClassDB::bind_method(D_METHOD("feed_current_state"), &RafkoGlue::feed_current_state);
  ClassDB::bind_method(D_METHOD("feed_next_state"), &RafkoGlue::feed_next_state);
  ClassDB::bind_method(D_METHOD("feed_consequences"), &RafkoGlue::feed_consequences);
}

bool RafkoGlue::configure_network(int input_size, PackedInt32Array layer_numbers, double expected_input_range) {

  /* Build Transfer_functions and layer sizes layers */
  std::vector<unsigned int> layer_structure(layer_numbers.size());
  std::vector<std::set<rafko_net::Transfer_functions>> layer_transfers(
    layer_numbers.size(), {rafko_net::transfer_function_selu}
  );

  for(int i=0; i<layer_numbers.size();++i){
    layer_structure[i] = layer_numbers[i];
  }

  rafko_net::RafkoNetBuilder network_builder = rafko_net::RafkoNetBuilder(*m_settings)
  .input_size(input_size)
  .expected_input_range(std::abs(expected_input_range))
  .allowed_transfer_functions_by_layer(layer_transfers);
  if(m_networkPtr)
    m_networkPtr = network_builder.create_layers(layer_structure);
    else network_builder.build_create_layers_and_swap(m_networkPtr, layer_structure);

  m_solverDeprecated = true;

  if(m_environment){
    bool retval = network_matches_environment();
    if(!retval)log_error("Network and environment sizes don't match!");
    return retval;
  }
  return true;
}

bool RafkoGlue::configure_environment(
  int state_size, double state_mean, double state_stddev,
  int action_size, double action_mean, double action_stddev
){
  m_environment = std::make_unique<Environment>(
    *this, state_size, state_mean, state_stddev,
    action_size, action_mean, action_stddev
  );
  if(m_networkPtr){
    bool retval = network_matches_environment();
    if(!retval)log_error("Network and environment sizes don't match!");
    return retval;
  }
  return static_cast<bool>(m_environment);
}


bool RafkoGlue::configure_trainer(int action_count, int q_set_size){
  if(!network_matches_environment()){
    log_error("Network and environment sizes don't match!");
    return false;
  }
  m_trainer = std::make_unique<rafko_gym::RafQTrainer>(
    *m_networkPtr, action_count, q_set_size, m_environment, m_objective, m_settings
  );
  return true;
}

PackedFloat32Array RafkoGlue::calculate(PackedFloat32Array network_input, bool reset){
  if(!m_networkPtr){
    log_error("Trying to calculate a non-existing network!");
    return PackedFloat32Array();
  }

  if(network_input.size() != static_cast<std::int32_t>(m_networkPtr->input_data_size())){
    log_error(
      "Input size mismatch! Network: " + std::to_string(m_networkPtr->input_data_size())
      + "; vs provided Input: " + std::to_string(network_input.size())
    );
    return PackedFloat32Array();
  }

  if(m_solverDeprecated){
    m_agent = rafko_net::SolutionSolver::Factory(*m_networkPtr, m_settings).build();
    m_solverDeprecated = false;
  }

  return toPoolArray(m_agent->solve(toStdVec(network_input), reset).acquire());
}

RafkoGlue::Environment::MaybeFeatureVector RafkoGlue::Environment::current_state() const{
  PackedFloat32Array data = m_parent.feed_current_state();
  if(0 == data.size())
    return {};
    else{
      m_currentStateBuffer = toStdVec(data);
      return {m_currentStateBuffer};
    }
}

RafkoGlue::Environment::StateTransition RafkoGlue::Environment::next(FeatureView action){
  Dictionary data = m_parent.feed_next_state(toPoolArray(action.acquire()));
  MaybeFeatureVector next_state;
  if(0 < static_cast<PackedFloat32Array>(data["state"]).size()){
    m_currentStateBuffer = toStdVec(static_cast<PackedFloat32Array>(data["state"]));
    next_state.emplace(m_currentStateBuffer);
  }
  return {next_state, static_cast<double>(data["q-value"]), static_cast<bool>(data["terminal"])};
}

RafkoGlue::Environment::StateTransition RafkoGlue::Environment::next(FeatureView state, FeatureView action) const{
  Dictionary data = m_parent.feed_consequences(toPoolArray(state.acquire()), toPoolArray(action.acquire()));
  MaybeFeatureVector next_state;
  if(0 < static_cast<PackedFloat32Array>(data["state"]).size()){
    m_queryStateBuffer = toStdVec(static_cast<PackedFloat32Array>(data["state"]));
    next_state.emplace(m_queryStateBuffer);
  }
  return {next_state, static_cast<double>(data["q-value"]), static_cast<bool>(data["terminal"])};
}

