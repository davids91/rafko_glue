extends HSlider

const min_val = 2e-10
const max_val = 0.02

func set_lr(new_value):
	var ob = get_parent().get_parent().get_node("RafkoGlue")
	var at = (100 - new_value) / 100
	var new_lr = min_val * at + max_val * (1.0 - at)
	get_parent().get_node("LRValueLabel").set_text(str(new_lr))
	ob.set_learning_rate(new_lr)
		
# Called when the node enters the scene tree for the first time.
func _ready():
	set_lr(get_value())

func _value_changed(new_value):
	set_lr(new_value)
