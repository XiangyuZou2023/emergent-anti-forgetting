-- Check what core provides
#check (· > · : Rat → Rat → Prop)
#check ((· + ·) : Rat → Rat → Rat)
#check ((· * ·) : Rat → Rat → Rat)
#check ((· - ·) : Rat → Rat → Rat)
#check Rat.add_comm
#check Rat.mul_comm
#check Rat.add_assoc
#check Rat.mul_assoc
-- Check if we have ordered ring lemmas
#check add_lt_add_right
#check mul_lt_mul_of_pos_right
-- Check simp
#check Nat.zero_lt_succ
