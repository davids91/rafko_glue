extends RafkoGlue

const legs = 2
const leg_segments = 2
const bodyparts = 1 + legs * leg_segments 
const joint_count = legs * 2
const actions_for_state = 2
const action_policy_size = (joint_count + 1) * actions_for_state
const state_policy_size = 5 + ((bodyparts-1) * 6)
const action_delay_ms = 50
const q_set_size = 5000
const expected_max_y = 500
const expected_max_linear_velocity = 1500
const expected_max_angular_velocity = 20
const action_range = 20

var expected_max_bodypart_distance
var start_state = []
var jail_state = []
var temp_state = []

var horse
var test_horse

func set_state(hors, state):
	var bodies = hors.get_children()
	var i = 0
	var bodpos
	for bodypart in bodies:
		if(bodypart.is_in_group("bodies")):
			var pos
			var ofs
			if(bodpos == null):
				pos = Vector2( \
				bodypart.get_global_position().x, \
				state[i + 0] * expected_max_y \
				)
				bodpos = pos
				ofs = 1
			else:
				pos = Vector2( \
				bodpos.x + (state[i + 0] * expected_max_bodypart_distance), \
				bodpos.y + (state[i + 1] * expected_max_bodypart_distance) \
				)
				ofs = 2
			var velo = Vector2( \
			state[i + ofs + 0] * expected_max_linear_velocity, \
			state[i + ofs + 1] * expected_max_linear_velocity \
			)
			var angle = state[i + ofs + 2] * (2*PI)
			var angvelo = state[i + ofs + 3] * expected_max_angular_velocity
			bodypart.reset_body(pos, angle, velo, angvelo)
			i = i + ofs + 4
	OS.delay_msec(action_delay_ms)

func get_state(hors):
	var state = []
	var bodies = hors.get_children()
	var bod
	for bodypart in bodies:
		if(bodypart.is_in_group("bodies")):
			if bod == null:
				bod = bodypart
				state.push_back(bodypart.get_global_position().y / expected_max_y)
			else:	
				state.push_back((bodypart.get_global_position().x - bod.get_global_position().x) / expected_max_bodypart_distance)
				state.push_back((bodypart.get_global_position().y - bod.get_global_position().y) / expected_max_bodypart_distance)
			state.push_back(bodypart.get_linear_velocity().x / expected_max_linear_velocity)
			state.push_back(bodypart.get_linear_velocity().y / expected_max_linear_velocity)
			state.push_back(bodypart.get_global_rotation() / (2*PI))
			state.push_back(bodypart.get_angular_velocity() / expected_max_angular_velocity)
	return state

func apply_action(hors, action):
#	print("applying action: ", action)
#	var mx = abs(action[0])
#	for d in action:
#		if mx < abs(d):
#			mx = abs(d)
#	print("max action: ", mx)
	var bodies = hors.get_children()
	var i = 0
	for body in bodies:
		for child in body.get_children():
			if(child.is_in_group("joints")):
#				print(child, "impulse: ", action[i], " / " , action.size())
				child._accept_impulse(action[i] * action_range)
				i = i + 1

func progress_callback(progress, step):
	print(\
	"progress: ", (progress * 100), ";\t", \
	"step: ", step, ";\t", \
	"error:", full_evaluation(true)\
	)
	if(0 == step): #init
		set_state(horse, temp_state)
	if(3 == step): #before optimizer build and training
		temp_state = get_state(horse)
		
func reset_environment():
	print("Resetting env..")	
	set_state(test_horse, jail_state)
	set_state(horse, start_state)

func feed_current_state():
	var current_state = Dictionary()	
	current_state["state"] = get_state(horse)

	var ob = horse.get_node("Body")
	var y = -ob.get_global_position().y
	var q_value = -ob.get_linear_velocity().x  / expected_max_linear_velocity
	q_value = q_value - (y - max(y, 150.0)) / 150.0

	current_state["terminal"] = (y < 80)
	current_state["q-value"] = q_value
	return current_state

func feed_next_state(action):
	apply_action(horse, action)
	OS.delay_msec(action_delay_ms)

	var ob = horse.get_node("Body")
	var y = -ob.get_global_position().y
	var q_value = -ob.get_linear_velocity().x  / expected_max_linear_velocity
	q_value = q_value - (y - max(y, 150.0)) / 150.0

	var result_state = Dictionary()	
	result_state["state"] = feed_current_state()
	result_state["terminal"] = (y < 80)
	result_state["q-value"] = q_value
#	print(\
#	"(1)state ", y, ";",\
#	"terminal: ", result_state["terminal"] , ";", \
#	"qval: " , result_state["q-value"] , ";" \
#	)

	if(result_state["terminal"]):
		temp_state = start_state
		result_state["q-value"] = 0.0
		OS.delay_msec(action_delay_ms)

	return result_state
	
func feed_consequences(state, action):
	set_state(horse, jail_state)
	set_state(test_horse, state)
	OS.delay_msec(action_delay_ms)

	apply_action(test_horse, action)
	OS.delay_msec(action_delay_ms)

	var ob = horse.get_node("Body")
	var y = -ob.get_global_position().y
	var q_value = -ob.get_linear_velocity().x  / expected_max_linear_velocity
	q_value = q_value - (y - max(y, 150.0)) / 150.0
	
	var consequence_state = Dictionary()
	consequence_state["state"] = get_state(test_horse)
	consequence_state["terminal"] = (y < 80)
	consequence_state["q-value"] =  q_value
	
	if(consequence_state["terminal"]):
		consequence_state["q-value"] = 0.0
		
#	print("cons terminal?", consequence_state["terminal"])
	set_state(test_horse, jail_state)
	set_state(horse, state)
#	print("cstate q-value: ", consequence_state["q-value"])
	OS.delay_msec(action_delay_ms)
	return consequence_state
	
func init_simu():
#	OS.delay_msec(1500)
	var ob = horse.get_node("Body")
	var ob2 = horse.get_node("LowerBackLeg")
	expected_max_bodypart_distance = (ob.get_global_position() - ob2.get_global_position()).length()
	expected_max_bodypart_distance = expected_max_bodypart_distance * 2
	start_state = get_state(horse)
	jail_state = get_state(test_horse)
	temp_state = start_state
	if !configure_network(state_policy_size, [10, action_policy_size], 1):
		print("Unable to configure network!")
		
	configure_env_and_trainer()

	print(get_latest_error())
	
func configure_env_and_trainer():
	if !configure_environment(state_policy_size, 0.5, 0.5, joint_count, 0.0, 1.0):
		print("Unable to configure Environment!")
				
	if !configure_trainer(actions_for_state, q_set_size):
		print("Unable to configure trainer!")
	
var delayed_start_thread
#Input: position, rotation and velocities for each leg; body position without x
#output: 4 joint impulse values + q value est
func _ready(): 
	horse = get_parent().get_node("Horse")
	test_horse = get_parent().get_node("testHorse")
	delayed_start_thread = Thread.new()
	var call_this = Callable(self, "init_simu")
	delayed_start_thread.start(call_this)
			
func _exit_tree():
	delayed_start_thread.wait_to_finish()
