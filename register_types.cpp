#include "register_types.h"

#include "core/class_db.h"
#include "rafko_glue.h"

void register_rafko_glue_types() {
  ClassDB::register_class<RafkoGlue>();
}

void unregister_rafko_glue_types() {
   // Nothing to do here in this example.
}
