-- Check what order ring lemmas core provides
#check mul_pos
#check mul_pos_of_pos_of_pos
#check sub_pos
#check sub_pos_of_lt
#check lt_of_lt_of_le
#check le_of_lt
#check add_comm
#check add_left
#check mul_comm
#check mul_left_comm
#check mul_assoc
#check add_lt_add_left
#check mul_lt_mul_of_pos_left
#check mul_lt_mul_of_pos_right
-- Check for `apply`-based proof
#check (fun (a b : Rat) (h : a > b) => ?_)
-- Available instances
#check (inst : LinearOrderedField Rat)
