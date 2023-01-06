extends Button

var ob

func _ready():
	ob = get_parent().get_parent().get_node("RafkoGlue")

func _pressed():
	print("inputs:")
	for i in ob.get_q_set_size():
		print(ob.get_q_set_input(i))
	print("=========================\nlabels:")
	for i in ob.get_q_set_size():
		print(ob.get_q_set_label(i))
#	var ob = get_parent().get_parent().get_node("Horse").get_node("Body")	
#	if ob.get_process_mode() == PROCESS_MODE_DISABLED:
#		ob.set_process_mode(PROCESS_MODE_PAUSABLE)
#	else:
#		ob.set_process_mode(PROCESS_MODE_DISABLED)
