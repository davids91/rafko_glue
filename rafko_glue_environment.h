#ifndef RAFKO_GLUE_ENVIRONMENT_H
#define RAFKO_GLUE_ENVIRONMENT_H

#include "rafko_glue.h"
#include "rafko_gym/services/rafko_agent.h"
#include "rafko_gym/services/rafko_environment.h"

class RafkoGlue;
class RafkoGlueEnvironment : public rafko_gym::RafkoEnvironment {
  public:
    RafkoGlueEnvironment(RafkoGlue& parent_)
    : parent(parent_)
    { }

    /*TODO#5: Implement prefill */
    sdouble32 full_evaluation(rafko_gym::RafkoAgent& agent);
    sdouble32 stochastic_evaluation(rafko_gym::RafkoAgent& agent, uint32 seed = 0u);
    sdouble32 get_training_fitness(void);
    sdouble32 get_testing_fitness(void);
    void push_state(void);
    void pop_state(void);

    void set_evaluation_parameters(int full_run_loops_, int stochastic_run_loops_){
      full_run_loops = full_run_loops_;
      stochastic_run_loops = stochastic_run_loops_;
    }

  private:
    RafkoGlue& parent;
    int stochastic_run_loops = 15;
    int full_run_loops = 100;
};


#endif /* RAFKO_GLUE_ENVIRONMENT_H */
