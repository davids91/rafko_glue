#include "rafko_glue_environment.h"

#include "rafko_utilities/models/const_vector_subrange.h"

sdouble32 RafkoGlueEnvironment::full_evaluation(rafko_gym::RafkoAgent& agent){
  pop_state(); push_state(); /* restore start state, and save it */
  sdouble32 fitness = parent.get_current_fitness();
  for(int i=0; i<full_run_loops; ++i){
    std::vector<double> vec = RafkoGlue::toStdVec(parent.provide_current_input_values());
    const rafko_utilities::ConstVectorSubrange<> net_output = agent.solve(
      vec,
      /* reset the data or not? */(i == 0),/*thread_index:*/0
    );
    parent.apply_network_output(RafkoGlue::toPoolArray({
      net_output.begin(), net_output.end(),
    }));
  }
  fitness = parent.get_current_fitness() - fitness; /* return the delta */
  pop_state();
  return fitness;
}

sdouble32 RafkoGlueEnvironment::stochastic_evaluation(rafko_gym::RafkoAgent& agent, uint32 seed){
  sdouble32 fitness = parent.get_current_fitness();
  for(int i=0; i<stochastic_run_loops; ++i){
    const rafko_utilities::ConstVectorSubrange<> net_output = agent.solve(
      RafkoGlue::toStdVec(parent.provide_current_input_values()),
      /* reset the data or not? */(i == 0),/*thread_index:*/0
    );
    parent.apply_network_output(RafkoGlue::toPoolArray({
      net_output.begin(), net_output.end(),
    }));
  }
  fitness = parent.get_current_fitness();
  return fitness;
}

sdouble32 RafkoGlueEnvironment::get_training_fitness(void){
  return 0;
}

sdouble32 RafkoGlueEnvironment::get_testing_fitness(void){
  return 0;
}

void RafkoGlueEnvironment::push_state(void){
  parent.push_state();
}

void RafkoGlueEnvironment::pop_state(void){
  parent.pop_state();
}
