extends Button

var thread = Thread.new()
var apply = false
var hors
var ob

func _ready():
	hors = get_parent().get_parent().get_node("Horse")
	ob = get_parent().get_parent().get_node("RafkoGlue")
	thread = Thread.new()
	var call_this = Callable(self, "apply_action")
	thread.start(call_this)

#TODO: switch behavior for action
func apply_action():
	Thread.set_thread_safety_checks_enabled(false)
	var rng = RandomNumberGenerator.new()
	rng.randomize()
#	ob.reset_environment()
	while true:
		if apply:
			var action = []
		#	print("Latest error: \n", ob.get_latest_error())
			var nn_result = ob.calculate(ob.feed_current_state()["state"], false)
	#		print(ob.get_latest_error())
			for i in 4:
				#action.push_back(rng.randf_range(-1, 1))	
				action.push_back(nn_result[i])	
			ob.apply_action(hors, action)
			OS.delay_msec(50)
			var t = str(hors.get_node("Body").get_position().x) + " m"
			get_parent().get_node("ProgressLabel").set_text(t)
			#get_parent().get_node("DebugLabel").set_text(ob.get_anlge_modif() + "")
			#get_parent().get_node("DebugLabel").set_text("asd")


func _pressed():
	apply = not apply
	if apply:
		set_text("Stop testing")
	else:
		set_text("Start testing")
	
func _exit_tree():
	thread.wait_to_finish()
