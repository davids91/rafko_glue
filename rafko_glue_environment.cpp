#include "rafko_glue_environment.h"

#include "rafko_utilities/models/const_vector_subrange.h"

void RafkoGlueEnvironment::notify_actions_processed(){
  std::lock_guard<std::mutex> my_lock(processed_feedback_mutex);
  feedback_processed = true;
  synchroniser.notify_one(); /* Only one thread is waiting for this */
}

sdouble32 RafkoGlueEnvironment::evaluation_function(rafko_gym::RafkoAgent& agent, int loops_to_do){
  std::cout << "[e]-->\t eval function started!" << std::endl;
  sdouble32 fitness = parent.get_current_fitness();
  for(int i=0; i<loops_to_do; ++i){
    std::vector<double> vec = RafkoGlue::toStdVec(parent.provide_current_input_values());
    const rafko_utilities::ConstVectorSubrange<> net_output = agent.solve(
      vec,
      /* reset the data or not? */(i == 0),/*thread_index:*/0
    );
    parent.apply_network_output(RafkoGlue::toPoolArray({
      net_output.begin(), net_output.end(),
    }));
    std::unique_lock<std::mutex> my_lock(processed_feedback_mutex);
    feedback_processed = false;
    synchroniser.wait(my_lock, [&](){return feedback_processed;});
    std::cout << "~" << std::flush;
  }
  std::cout << std::endl;
  fitness = parent.get_current_fitness() - fitness; /* return the delta */
  std::cout << "[e]-->\t Fitness from "<< loops_to_do <<" loops:  " << fitness << std::endl;
  return fitness;
}

sdouble32 RafkoGlueEnvironment::full_evaluation(rafko_gym::RafkoAgent& agent){
  sdouble32 current_fitness = evaluation_function(agent, full_run_loops);
  testing_fitness = (testing_fitness + current_fitness)/2.0;
  return current_fitness;
}

sdouble32 RafkoGlueEnvironment::stochastic_evaluation(rafko_gym::RafkoAgent& agent, uint32 seed){
  sdouble32 current_fitness = evaluation_function(agent, stochastic_run_loops);
  training_fitness = (training_fitness + current_fitness)/2.0;
  return current_fitness;
}

sdouble32 RafkoGlueEnvironment::get_training_fitness(void){
  return training_fitness;
}

sdouble32 RafkoGlueEnvironment::get_testing_fitness(void){
  return testing_fitness;
}

void RafkoGlueEnvironment::notify_pop_processed(){
  std::lock_guard<std::mutex> my_lock(processed_pop_mutex);
  pop_processed = true;
  pop_synchroniser.notify_one(); /* Only one thread is waiting for this */
}

void RafkoGlueEnvironment::push_state(void){
  parent.push_state();
}

void RafkoGlueEnvironment::pop_state(void){
  parent.pop_state();
  {
    std::unique_lock<std::mutex> my_lock(processed_pop_mutex);
    std::cout << "[e]-->\t Pop state waiting to notify!" << std::endl;
    pop_processed = false;
    pop_synchroniser.wait(my_lock, [&](){return pop_processed;});
    std::cout << "[e]-->\t Pop state notified!" << std::endl;
  }
}
