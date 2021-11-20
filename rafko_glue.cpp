#include "rafko_glue.h"

#include <vector>

#include "rafko_net/services/rafko_net_builder.h"

void RafkoGlue::_bind_methods() {
  ClassDB::bind_method(D_METHOD("optimize_step"), &RafkoGlue::optimize_step);
  ClassDB::bind_method(D_METHOD("create_network"), &RafkoGlue::create_network);
  ClassDB::bind_method(D_METHOD("get_env"), &RafkoGlue::get_env);
  ClassDB::bind_method(D_METHOD("step"), &RafkoGlue::step);
}

bool RafkoGlue::create_network(int input_size, PoolVector<int> layer_numbers) {
  if(network)network.reset();

  /* Build Transfer_functions and layer sizes layers */
  std::vector<unsigned int> layer_structure(layer_numbers.size());
  std::vector<std::vector<rafko_net::Transfer_functions>> layer_transfers(
    layer_numbers.size(), {rafko_net::transfer_function_selu}
  );
  for(int i=0; i<layer_numbers.size();++i){
    layer_structure[i] = layer_numbers[i];
  }

  network = std::unique_ptr<rafko_net::RafkoNet>(
    rafko_net::RafkoNetBuilder(context).input_size(input_size)
    .allowed_transfer_functions_by_layer(layer_transfers)
    .dense_layers(layer_structure)
  );

  optimizer = std::make_unique<rafko_gym::RafkoNetApproximizer>(
    context, *network, *environment, rafko_net::weight_updater_amsgrad
  );
  return true;
}

PoolRealArray RafkoGlue::get_env(){
  std::cout << "get_env called!" << std::endl;
  return PoolRealArray();
}

void RafkoGlue::step(PoolRealArray net_output){
  std::cout << "step called with:";
  for(int i=0; i<net_output.size(); ++i){
    std::cout << "["<< net_output[i] <<"]";
  }
  std::cout << std::endl;
}
