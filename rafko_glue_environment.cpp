#include "rafko_glue_environment.h"

sdouble32 RafkoGlueEnvironment::full_evaluation(rafko_gym::RafkoAgent& RafkoAgent){
  std::cout << "Full evaluation called!" << std::endl;
  return 0;
}

sdouble32 RafkoGlueEnvironment::stochastic_evaluation(rafko_gym::RafkoAgent& RafkoAgent, uint32 seed){
  std::cout << "Stochastic evaluation called!" << std::endl;
  Variant ret = parent.provide_current_input_values();
  // for(int i=0; i<5; ++i){
  //   const rafko_utilities::DataRingbuffer& output = RafkoAgent.solve(get_env(),(i == 0),/*thread_index:*/0);
  //   step(output.get_const_element(0)); /* This contains not only the output but the Neuron data */
  // }
  /* evaluate RafkoAgent */
  PoolRealArray net_output;
  net_output.push_back(0.1);
  net_output.push_back(0.3);
  net_output.push_back(0.5);

  parent.apply_network_output(net_output);
  return 0;
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
