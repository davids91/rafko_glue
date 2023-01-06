extends RigidBody2D

var reset = false
var reset_to_position = Vector2()
var reset_to_velo = Vector2()
var reset_to_angvelo = 0.0
var reset_to_rotation = 0.0
func _integrate_forces(state):
	if(reset):
		state.set_angular_velocity(reset_to_angvelo)
		state.set_linear_velocity(reset_to_velo)
		state.set_transform(Transform2D(reset_to_rotation, reset_to_position))
		reset = false
		
func reset_body(pos, rot, velo, angvelo):
	reset_to_position = pos
	reset_to_rotation = rot
	reset_to_velo = velo
	reset_to_angvelo = angvelo
	reset = true
