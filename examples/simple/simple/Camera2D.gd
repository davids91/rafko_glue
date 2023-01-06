extends Camera2D


# Called when the node enters the scene tree for the first time.
func _ready():
	pass # Replace with function body.

func _input(event):
	if event is InputEventMouseButton:
		if event.is_pressed():
			if event.button_index == MOUSE_BUTTON_WHEEL_UP:
				set_zoom(get_zoom() * 1.2)
			if event.button_index == MOUSE_BUTTON_WHEEL_DOWN:
				set_zoom(get_zoom() * 0.8)
				
func _process(_delta):
	set_global_position(get_parent().get_node("Horse").get_node("Body").get_global_position())

