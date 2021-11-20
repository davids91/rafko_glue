#ifndef RAFKO_GLUE_H
#define RAFKO_GLUE_H

#include "scene/main/node.h"

#include "rafko_protocol/common.pb.h"
#include "rafko_protocol/rafko_net.pb.h"
#include "rafko_gym/services/agent.h"
#include "rafko_gym/services/environment.h"
#include "rafko_gym/services/rafko_net_approximizer.h"
#include "rafko_mainframe/models/service_context.h"

#include <memory>

class RafkoGlue : public Node {
  GDCLASS(RafkoGlue, Node);

  class RafkoGlueEnvironment : public rafko_gym::Environment {
    public:
      RafkoGlueEnvironment(RafkoGlue& parent_)
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
  bool create_network(
    int input_size, PoolVector<int> layer_numbers
    /* TODO: PoolVector<rafko_net::Transfer_functions> layer_functions */
  );
  /* TODO: Load and save network */

  RafkoGlue(){
    environment = std::make_unique<RafkoGlueEnvironment>(*this);
  }
  ~RafkoGlue(){
    environment.reset();
    if(network)network.reset();
    if(optimizer)optimizer.reset();
  }
protected:
  static void _bind_methods();
private:
  rafko_mainframe::ServiceContext context;
  std::unique_ptr<RafkoGlueEnvironment> environment;
  std::unique_ptr<rafko_net::RafkoNet> network;
  std::unique_ptr<rafko_gym::RafkoNetApproximizer> optimizer;

};

#endif /* RAFKO_GLUE_H */
