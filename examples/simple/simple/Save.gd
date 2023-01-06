extends Button

var ob

func _ready():
	ob = get_parent().get_parent().get_node("RafkoGlue")

func _pressed():
	print("Saving Network..")
	ob.save_network()
	print(ob.get_latest_error())
