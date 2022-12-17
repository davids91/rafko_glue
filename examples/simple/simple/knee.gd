extends RigidBody2D

func _accept_impulse(amount):
	get_node("LowerLeg/kneejoint")._accept_impulse(amount)
	
# Called when the node enters the scene tree for the first time.
func _ready():
	pass # Replace with function body.


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta):
	pass
