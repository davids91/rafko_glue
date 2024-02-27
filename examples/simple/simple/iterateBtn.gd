extends Button

var thread = Thread.new()

func iterate_once():
	Thread.set_thread_safety_checks_enabled(false)
	var iteration_count = int(get_parent().get_node("IterationCount").get_text())
	var ob = get_parent().get_parent().get_node("RafkoGlue")

	for i in iteration_count:
#		ob.reset_environment() #until pausing is not possible
		var discovery_length = get_parent().get_node("ExploreSteps").get_text()
		var exploration_ratio = get_parent().get_node("exploreSlider").get_value()
		var training_epochs = get_parent().get_node("TrainingEpochs").get_text()
		ob.iterate(int(discovery_length), float(exploration_ratio), int(training_epochs))
		print(ob.get_latest_error())
		print("Iteration ", (i+1), "/" , iteration_count ," Done!", \
		" Error: ", ob.full_evaluation(true), \
		" Q-set size: ", ob.get_q_set_size()
		)
		ob.save_network()

func _pressed():
	if thread.is_started():	
		thread.wait_to_finish()
	thread = Thread.new()
	var call_this = Callable(self, "iterate_once")
	if not thread.is_started():
		thread.start(call_this)
	else:
		print ("Already on it boss!")
		
func _exit_tree():
	thread.wait_to_finish()
