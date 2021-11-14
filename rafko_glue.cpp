#include "rafko_glue.h"

void RafkoGlue::_bind_methods() {
  ClassDB::bind_method(D_METHOD("get_env"), &RafkoGlue::get_env);
  ClassDB::bind_method(D_METHOD("step"), &RafkoGlue::step);
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
