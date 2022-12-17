extends Node2D


# Called when the node enters the scene tree for the first time.
func _ready():
	pass # Replace with function body.

func _input(event):
	if(Input.is_key_pressed(KEY_A)):
		get_node("backLeg")._accept_impulse(10)
	if(Input.is_key_pressed(KEY_D)):
		get_node("backLeg")._accept_impulse(-10)
	if(event.is_action_pressed("ui_left")):
		get_node("frontLeg")._accept_impulse(10)
	if(event.is_action_pressed("ui_right")):
		get_node("frontLeg")._accept_impulse(-10)

# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta):
	pass
