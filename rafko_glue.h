#ifndef RAFKO_GLUE_H
#define RAFKO_GLUE_H

#include "scene/main/node.h"

#include <memory>
#include <string>

#include "rafko_protocol/rafko_net.pb.h"
#include "rafko_protocol/training.pb.h"
#include "rafko_mainframe/models/rafko_autonomous_entity.hpp"
#include "rafko_gym/models/rafq_environment.hpp"
#include "rafko_gym/models/rafko_objective.hpp"
#include "rafko_gym/models/rafko_cost.hpp"
#include "rafko_gym/services/rafq_trainer.hpp"
#include "rafko_net/services/solution_solver.hpp"


// TODO: set Objective
// TODO: include PoolVector<rafko_net::Transfer_functions> layer_functions
// TODO: Load and save network
class RafkoGlue : public Node, public rafko_mainframe::RafkoAutonomousEntity
{
  GDCLASS(RafkoGlue, Node);
  using RafQEnvironment = rafko_gym::RafQEnvironment;
  class Environment;
public:
  RafkoGlue() 
  : Node()
  , rafko_mainframe::RafkoAutonomousEntity()
  {
    m_settings->set_learning_rate(1e-4)
    .set_training_relevant_loop_count(5)
    .set_delta(0.0000000006);
  }

  ~RafkoGlue() = default;

  bool configure_trainer(int action_count, int q_set_size);
  bool configure_environment( //TODO: Document that env needs ONE action size, while network output is multiple actions size
    int state_size, double state_mean, double state_stddev,
    int action_size, double action_mean, double action_stddev
  );
  bool configure_network(int input_size, PackedInt32Array layer_numbers, double expected_input_range);

  void set_learning_rate(double learning_rate){
    m_settings->set_learning_rate(1e-4);
  }

  void save_network();
  void load_network();

  PackedFloat32Array calculate(PackedFloat32Array network_input, bool reset);

  double full_evaluation(bool force_gpu_upload = false){
    if(!m_trainer){
      log_error("Trying to evaluate a non-existent set!");
      return 0.0;
    }
    return m_trainer->full_evaluation(force_gpu_upload);
  }

  void progress_callback(double progress, int step){
    get_script_instance()->call(StringName("progress_callback"), progress, step);  
  }

  void iterate(int discovery_length, float exploration_ratio, int training_epochs){
    if(!network_matches_environment()){
      std::string detail_string;
      if(m_trainer && m_networkPtr)
        detail_string = (
          ": " + std::to_string(m_trainer->q_set().get_feature_size()) 
          + "<>" + std::to_string(m_networkPtr->output_neuron_number())
        );
        else detail_string = "!";
      log_error("Network and environment sizes don't match" + detail_string);
      return;
    }

    if(!m_trainer){
      log_error("Tried to use a non-existing trainer!");
      return;
    }

    m_trainer->iterate(
      discovery_length, exploration_ratio, training_epochs,
      [this](double progress, std::uint32_t step){ progress_callback(progress, step); }
    );
    m_solverDeprecated = true;
  } 

  StringName get_latest_error(){
    return StringName(m_lastErrorMessage.c_str());
  }

  int get_q_set_size(){
    if(!m_trainer){
      log_error("Trying to query a non-existent set!");
      return 0.0;
    }
    return m_trainer->q_set_size();
  }

  PackedFloat32Array get_q_set_input(int index) const;
  PackedFloat32Array get_q_set_label(int index) const;

  void reset_environment(){
    get_script_instance()->call(StringName("reset_environment"));    
  }

  PackedFloat32Array feed_current_state() const{ // might return with a vector of size 0
    return get_script_instance()->call(StringName("feed_current_state"));    
  }

  Dictionary feed_next_state(PackedFloat32Array action){ // needs to return with an array, a q value and a terminal flag
    return get_script_instance()->call(StringName("feed_next_state"), action);
  }
  
  Dictionary feed_consequences(PackedFloat32Array state, PackedFloat32Array action) const{
    return get_script_instance()->call(StringName("feed_consequences"), state, action);        
  }

protected:
  static void _bind_methods();

  void log_error(std::string err) const{
    ++m_errorCount;
    m_lastErrorMessage = "error[" + std::string(std::to_string(m_errorCount)) + "]:" + err;
  }

private:
  rafko_net::RafkoNet* m_networkPtr = nullptr;
  std::shared_ptr<RafQEnvironment> m_environment;
  std::shared_ptr<rafko_gym::RafkoObjective> m_objective = std::make_shared<rafko_gym::RafkoCost>(
    *m_settings, rafko_gym::cost_function_mse
  );  
  std::unique_ptr<rafko_gym::RafQTrainer> m_trainer;
  std::shared_ptr<rafko_net::SolutionSolver> m_agent;
  bool m_solverDeprecated = true;
  mutable std::string m_lastErrorMessage;
  mutable int m_errorCount = 0;

  bool network_matches_environment() const{
    return (
      static_cast<bool>(m_environment) && m_networkPtr
      &&(m_environment->state_size() == m_networkPtr->input_data_size())
      &&(
        (!m_trainer && (m_environment->action_size() == m_networkPtr->output_neuron_number()))
        ||(m_trainer && (m_trainer->q_set().get_feature_size() == m_networkPtr->output_neuron_number()))
      )
    );
  }

  class Environment : public RafQEnvironment{
  public:
    Environment(
      RafkoGlue& parent,
      int state_size, double state_mean, double state_stddev,
      int action_size, double action_mean, double action_stddev
    )
    : RafQEnvironment(state_size, action_size, {state_mean, state_stddev}, {action_mean, action_stddev})
    , m_parent(parent)
    {

    }
    void reset() override{
      m_parent.reset_environment();
    }
    MaybeFeatureVector current_state() const override;
    StateTransition next(FeatureView action) override;
    StateTransition next(FeatureView state, FeatureView action) const override;
  private:
    RafkoGlue& m_parent;
    mutable FeatureVector m_currentStateBuffer; //TODO: Guard against multithreaded access?
    mutable FeatureVector m_queryStateBuffer;
  };
};

#endif /* RAFKO_GLUE_H */
