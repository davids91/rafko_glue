#ifndef RAFKO_GLUE_H
#define RAFKO_GLUE_H

#include "scene/main/node.h"

#include "rafko_gym/services/agent.h"
#include "rafko_gym/services/environment.h"

#include <memory>

class RafkoGlue : public Node {
  GDCLASS(RafkoGlue, Node);

  class RafkoEnvironment : public Environment { /* why not rafko_gym::Environment ??? */
    public:
      RafkoEnvironment(RafkoGlue& parent_)
      : parent(parent_){ }
      sdouble32 full_evaluation(rafko_gym::Agent& agent){
        std::cout << "Full evaluation called!" << std::endl;
        return 0;
      }
      sdouble32 stochastic_evaluation(rafko_gym::Agent& agent, uint32 seed = 0u){
        Variant ret = parent.get_script_instance()->call(StringName("get_env"));
        // for(int i=0; i<5; ++i){
        //   const rafko_utilities::DataRingbuffer& output = agent.solve(get_env(),(i == 0),/*thread_index:*/0);
        //   step(output.get_const_element(0)); /* This contains not only the output but the Neuron data */
        // }
        /* evaluate agent */
        PoolRealArray net_output;
        net_output.push_back(0.1);
        net_output.push_back(0.3);
        net_output.push_back(0.5);

        parent.get_script_instance()->call(StringName("step"), net_output);
        std::cout << "Stochastic evaluation called!" << std::endl;
        return 0;
      }
      void push_state(void){std::cout << "push state called!" << std::endl;}
      sdouble32 get_training_fitness(void){
        return 0;
      }
      sdouble32 get_testing_fitness(void){
        return 0;
      }
      void pop_state(void){}
    private:
      RafkoGlue& parent;
  };

public:
  void _init() {}

  PoolRealArray get_env(); /* Provides input value for the network */
  void step(PoolRealArray net_output); /* Apply network output to the environment */

  RafkoGlue(){
    environment = std::make_unique<RafkoEnvironment>(*this);
  }
  ~RafkoGlue(){
    environment.reset();
  }
protected:
  static void _bind_methods();
private:
  std::unique_ptr<RafkoEnvironment> environment;
};

#endif /* RAFKO_GLUE_H */
