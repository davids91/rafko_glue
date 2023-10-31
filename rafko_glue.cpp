#include "rafko_glue.h"

#include <vector>
#include <cmath>
#include <optional>
#include <iostream>
#include <fstream>

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
  ClassDB::bind_method(D_METHOD("get_latest_error"), &RafkoGlue::get_latest_error);
  ClassDB::bind_method(D_METHOD("full_evaluation"), &RafkoGlue::full_evaluation);
  ClassDB::bind_method(D_METHOD("get_q_set_size"), &RafkoGlue::get_q_set_size);
  ClassDB::bind_method(D_METHOD("get_q_set_label"), &RafkoGlue::get_q_set_label);
  ClassDB::bind_method(D_METHOD("get_q_set_input"), &RafkoGlue::get_q_set_input);
  ClassDB::bind_method(D_METHOD("set_learning_rate"), &RafkoGlue::set_learning_rate);
  ClassDB::bind_method(D_METHOD("save_network"), &RafkoGlue::save_network);
  ClassDB::bind_method(D_METHOD("load_network"), &RafkoGlue::load_network);
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

  if(!m_networkPtr)
    m_networkPtr = network_builder.create_layers(layer_structure);
    else network_builder.create_layers_and_swap(m_networkPtr, layer_structure);

  m_solverDeprecated = true;
  //TODO: handle if there's an environment or trainer already
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
  return static_cast<bool>(m_environment);
}


bool RafkoGlue::configure_trainer(int action_count, int q_set_size){
  std::size_t feature_size = 0;
  if(m_environment){
    feature_size = action_count * (m_environment->action_size() + 1);
  }
  if(feature_size != m_networkPtr->output_neuron_number()){
    log_error(
      "Network and environment sizes don't match: feature:" + std::to_string(feature_size) 
      + " vs network output:" + std::to_string(m_networkPtr->output_neuron_number())
    );
    return false;
  }
  m_trainer = std::make_unique<rafko_gym::RafQTrainer>(
    *m_networkPtr, action_count, q_set_size, m_environment, m_objective, m_settings
  );
  m_trainer->set_weight_updater(rafko_gym::weight_updater_amsgrad);
  return true;
}

void RafkoGlue::save_network(){
  if(!m_networkPtr){
    log_error("Can't save a non-existent Network!");
    return;
  }

  std::filebuf file_buffer;
  file_buffer.open("network.rfnet", std::ios::out);
  std::ostream os(&file_buffer);
  m_networkPtr->SerializeToOstream(&os);
  file_buffer.close();
}

void RafkoGlue::load_network(){
  std::filebuf file_buffer;
  file_buffer.open("network.rfnet", std::ios::in);

  if(!file_buffer.is_open()){
    log_error("Can't load a non-existent Network!");
    return;
  }
  
  std::istream is(&file_buffer);
  m_networkPtr->ParseFromIstream(&is);
  file_buffer.close();

  m_trainer.reset();
  m_solverDeprecated = true;
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

PackedFloat32Array RafkoGlue::get_q_set_input(int index) const{
  if(!m_trainer || static_cast<int>(m_trainer->q_set().get_number_of_input_samples()) <= index){
    log_error("Q set Unavailable or index out of bounds!");
    return PackedFloat32Array();
  }
  return toPoolArray(m_trainer->q_set().get_input_sample(index));
}

PackedFloat32Array RafkoGlue::get_q_set_label(int index) const{
  if(!m_trainer || static_cast<int>(m_trainer->q_set().get_number_of_label_samples()) <= index){
    log_error("Q set Unavailable or index out of bounds!");
    return PackedFloat32Array();
  }
  return toPoolArray(m_trainer->q_set().get_label_sample(index));
}

static rafko_gym::RafQEnvironment::FeatureVector s_buf;
RafkoGlue::Environment::StateTransition RafkoGlue::Environment::current_state() const{
  Dictionary data = m_parent.feed_current_state__();
  s_buf = toStdVec(static_cast<PackedFloat32Array>(data["state"]));
  MaybeFeatureVector state;
  state.emplace(s_buf);
  if(0 == data.size()) return {};
  return {state, static_cast<double>(data["q-value"]), static_cast<bool>(data["terminal"])};
}

RafkoGlue::Environment::StateTransition RafkoGlue::Environment::next(FeatureView action){
  Dictionary data = m_parent.feed_next_state__(toPoolArray(action.acquire()));
  MaybeFeatureVector next_state;
  if(0 < static_cast<PackedFloat32Array>(data["state"]).size()){
    m_currentStateBuffer = toStdVec(static_cast<PackedFloat32Array>(data["state"]));
    next_state.emplace(m_currentStateBuffer);
  }
  return {next_state, static_cast<double>(data["q-value"]), static_cast<bool>(data["terminal"])};
}

RafkoGlue::Environment::StateTransition RafkoGlue::Environment::next(FeatureView state, FeatureView action, const AnyData &user_data) const{
  Dictionary data = m_parent.feed_consequences__(toPoolArray(state.acquire()), toPoolArray(action.acquire()));
  MaybeFeatureVector next_state;
  if(0 < static_cast<PackedFloat32Array>(data["state"]).size()){
    m_queryStateBuffer = toStdVec(static_cast<PackedFloat32Array>(data["state"]));
    next_state.emplace(m_queryStateBuffer);
  }
  return {next_state, static_cast<double>(data["q-value"]), static_cast<bool>(data["terminal"])};
}

