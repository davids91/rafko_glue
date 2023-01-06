#include "register_types.h"

#include "core/object/class_db.h"
#include "rafko_glue.h"

void initialize_rafko_glue_module(ModuleInitializationLevel p_level) {
  if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
    return;
  }
  ClassDB::register_class<RafkoGlue>();
}

void uninitialize_rafko_glue_module(ModuleInitializationLevel p_level) {
  if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
    return; 
  }
}