#ifndef RAFKO_GLUE_ENVIRONMENT_H
#define RAFKO_GLUE_ENVIRONMENT_H

#include "rafko_glue.h"
#include "rafko_gym/services/rafko_agent.h"
#include "rafko_gym/services/rafko_environment.h"

class RafkoGlue;
class RafkoGlueEnvironment : public rafko_gym::RafkoEnvironment {
  public:
    RafkoGlueEnvironment(RafkoGlue& parent_)
    : parent(parent_){ }

    sdouble32 full_evaluation(rafko_gym::RafkoAgent& RafkoAgent);
    sdouble32 stochastic_evaluation(rafko_gym::RafkoAgent& RafkoAgent, uint32 seed = 0u);
    sdouble32 get_training_fitness(void);
    sdouble32 get_testing_fitness(void);
    void push_state(void);
    void pop_state(void);

  private:
    RafkoGlue& parent;
};


#endif /* RAFKO_GLUE_ENVIRONMENT_H */
