from libc.stdio cimport printf

from omegaconf import OmegaConf

from mettagrid.action_handler cimport ActionArg
from mettagrid.actions.metta_action_handler cimport MettaActionHandler

from mettagrid.grid_object cimport (
    GridLocation,
    GridObject,
    GridObjectId,
    Orientation
)
from mettagrid.objects.agent cimport Agent

cdef class Move(MettaActionHandler):
    def __init__(self, cfg: OmegaConf):
        MettaActionHandler.__init__(self, cfg, "move")

    cdef unsigned char max_arg(self):
        return 1

    cdef bint _handle_action(
        self,
        unsigned int actor_id,
        Agent * actor,
        ActionArg arg):

        cdef unsigned short direction = arg

        cdef Orientation orientation = <Orientation>(actor.orientation)
        if direction == 1:
            if orientation == Orientation.Up:
                orientation = Orientation.Down
            elif orientation == Orientation.Down:
                orientation = Orientation.Up
            elif orientation == Orientation.Left:
                orientation = Orientation.Right
            elif orientation == Orientation.Right:
                orientation = Orientation.Left

        cdef GridLocation old_loc = actor.location
        cdef GridLocation new_loc = self.env._grid.relative_location(old_loc, orientation)
        if not self.env._grid.is_empty(new_loc.r, new_loc.c):
            return 0
        return self.env._grid.move_object(actor.id, new_loc)
