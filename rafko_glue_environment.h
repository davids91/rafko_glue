#ifndef RAFKO_GLUE_ENVIRONMENT_H
#define RAFKO_GLUE_ENVIRONMENT_H

#include <mutex>
#include <condition_variable>

#include "rafko_gym/services/rafko_agent.h"
#include "rafko_gym/services/rafko_environment.h"

#include "rafko_glue.h"

class RafkoGlue;
class RafkoGlueEnvironment : public rafko_gym::RafkoEnvironment {
  public:
    RafkoGlueEnvironment(RafkoGlue& parent_)
    : parent(parent_)
    { }

    /*TODO#5: Implement prefill */
    sdouble32 full_evaluation(rafko_gym::RafkoAgent& agent);
    sdouble32 stochastic_evaluation(rafko_gym::RafkoAgent& agent, uint32 seed = 0u);
    sdouble32 get_training_fitness();
    sdouble32 get_testing_fitness();
    void push_state();
    void pop_state();
    void notify_actions_processed();

    void set_evaluation_parameters(int full_run_loops_, int stochastic_run_loops_){
      full_run_loops = full_run_loops_;
      stochastic_run_loops = stochastic_run_loops_;
    }

  private:
    RafkoGlue& parent;
    int stochastic_run_loops = 15;
    int full_run_loops = 100;
    std::condition_variable synchroniser;
    std::mutex processed_feedback_mutex;
    bool feedback_processed = false;

    sdouble32 evaluation_function(rafko_gym::RafkoAgent& agent, int loops_to_do);

};


#endif /* RAFKO_GLUE_ENVIRONMENT_H */
