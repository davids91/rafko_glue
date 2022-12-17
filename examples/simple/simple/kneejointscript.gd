extends PinJoint2D

func _accept_impulse(amount): #TODO: apply force based on a circle ( center is the joint ) 
	var avg_pos = (get_node(node_a).get_global_position() + get_node(node_b).get_global_position())/2.0
	get_node(node_a).apply_central_impulse( - (get_node(node_a).get_global_position() - avg_pos) * amount)
	get_node(node_b).apply_central_impulse( - (get_node(node_b).get_global_position() - avg_pos) * amount)
	
#func _input(event):
#	if(event.is_action_pressed("ui_left")):
#		_accept_impulse(10)
#	if(event.is_action_pressed("ui_right")):
#		_accept_impulse(-10)

func _ready():
	pass
