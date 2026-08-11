from math import sqrt

def where_is_neighbor(my_velocity, direction_to_neighbor):
    """Determine the relative position of a neighbor based on velocity direction."""
    dot   = my_velocity[0]*direction_to_neighbor[0] + \
            my_velocity[1]*direction_to_neighbor[1]
    
    cross = my_velocity[0]*direction_to_neighbor[1] - \
            my_velocity[1]*direction_to_neighbor[0]
    
    if dot > 0 and cross > 0:
        print("AHEAD-LEFT")
    elif dot > 0 and cross < 0:
        print("AHEAD-RIGHT")
    elif dot < 0 and cross > 0:
        print("BEHIND-LEFT")
    elif dot < 0 and cross < 0:
        print("BEHIND-RIGHT")
    else:
        print("PERPENDICULAR or PARALLEL")

my_velocity = (1, 0)

where_is_neighbor(my_velocity, (0.9, 0.3))
where_is_neighbor(my_velocity, (0.9, -0.3))
where_is_neighbor(my_velocity, (-0.5, 0.4))
where_is_neighbor(my_velocity, (-0.5, -0.4))
