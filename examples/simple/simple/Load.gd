extends Button

var ob

func _ready():
	ob = get_parent().get_parent().get_node("RafkoGlue")

func _pressed():
	print("Loading Network..")
	ob.load_network()
	ob.configure_env_and_trainer()
