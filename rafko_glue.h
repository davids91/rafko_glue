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
  RafkoGlue() : Node(), rafko_mainframe::RafkoAutonomousEntity(){}
  ~RafkoGlue() = default;

  bool configure_trainer(int action_count, int q_set_size);
  bool configure_environment(
    int state_size, double state_mean, double state_stddev,
    int action_size, double action_mean, double action_stddev
  );
  bool configure_network(int input_size, PackedInt32Array layer_numbers, double expected_input_range);

  PackedFloat32Array calculate(PackedFloat32Array network_input, bool reset);

  void iterate(int discovery_length, float exploration_ratio, int training_epochs){
    if(!network_matches_environment()){
      log_error("Network and environment sizes don't match!");
      return;
    }

    if(!m_trainer){
      log_error("Tried to use a non-existing trainer!");
      return;
    }

    m_trainer->iterate(discovery_length, exploration_ratio, training_epochs);
    m_solverDeprecated = true;
  } 

  StringName get_latest_error(){
    return StringName(m_lastErrorMessage.c_str());
  }
  void reset_environment(){
    get_script_instance()->call(StringName("reset_environment"));    
  }

  PackedFloat32Array feed_current_state() const{ // might return with a vector of size 0
    return get_script_instance()->call(StringName("feed_current_state"));    
  }

  Dictionary feed_next_state(PackedFloat32Array action){ // needs to return with an array, a q value and a terminal flag
    return get_script_instance()->call(StringName("feed_next_state"));        
  }
  
  Dictionary feed_consequences(PackedFloat32Array state, PackedFloat32Array action) const{
    return get_script_instance()->call(StringName("feed_consequences"));        
  }

protected:
  static void _bind_methods();

  void log_error(std::string err){
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
  std::string m_lastErrorMessage;
  int m_errorCount = 0;

  bool network_matches_environment() const{
    return (
      static_cast<bool>(m_environment) && m_networkPtr
      &&(m_environment->state_size() == m_networkPtr->input_data_size())
      &&(m_environment->action_size() == m_networkPtr->output_neuron_number())
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
