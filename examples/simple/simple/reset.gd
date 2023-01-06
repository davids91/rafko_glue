extends Button

func _pressed():
	get_parent().get_parent().get_node("RafkoGlue").reset_environment()
	
