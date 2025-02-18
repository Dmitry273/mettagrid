
from libc.stdio cimport printf

from omegaconf import OmegaConf

from mettagrid.grid_object cimport GridLocation, Orientation
from mettagrid.action cimport ActionArg
from mettagrid.objects.agent cimport Agent
from mettagrid.objects.metta_object cimport MettaObject
from mettagrid.objects.constants cimport ObjectType, Events, GridLayer, ObjectTypeNames, InventoryItem
from mettagrid.objects.usable cimport Usable
from mettagrid.objects.generator cimport Generator
from mettagrid.objects.converter cimport Converter
from mettagrid.actions.actions cimport MettaActionHandler

cdef class Use(MettaActionHandler):
    def __init__(self, cfg: OmegaConf):
        MettaActionHandler.__init__(self, cfg, "use")

    cdef unsigned char max_arg(self):
        return 0

    cdef bint _handle_action(
        self,
        unsigned int actor_id,
        Agent * actor,
        ActionArg arg):

        cdef GridLocation target_loc = self.env._grid.relative_location(
            actor.location,
            <Orientation>actor.orientation
        )
        target_loc.layer = GridLayer.Object_Layer
        cdef MettaObject *target = <MettaObject*>self.env._grid.object_at(target_loc)
        if target == NULL or not target.is_usable_type():
            return False

        cdef Usable *usable = <Usable*> target

        if not usable.usable(actor):
            return False

        actor.update_energy(-usable.use_cost, &self.env._rewards[actor_id])

        usable.ready = 0
        self.env._event_manager.schedule_event(Events.Reset, usable.cooldown, usable.id, 0)

        actor.stats.incr(self._stats.target[target._type_id])
        actor.stats.incr(self._stats.target[target._type_id], actor.group_name)
        actor.stats.set_once(self._stats.target_first_use[target._type_id], self.env._current_timestep)

        actor.stats.add(self._stats.target_energy[target._type_id], usable.use_cost + self.action_cost)
        actor.stats.add(self._stats.target_energy[target._type_id], actor.group_name, usable.use_cost + self.action_cost)


        if target._type_id == ObjectType.AltarT:
            self.env._rewards[actor_id] += 1

        cdef Generator *generator
        if target._type_id == ObjectType.GeneratorT:
            generator = <Generator*>target
            generator.r1 -= 1
            actor.update_inventory(InventoryItem.r1, 1, &self.env._rewards[actor_id])
            self.env._stats.incr(b"r1.harvested")

        cdef Converter *converter
        if target._type_id == ObjectType.ConverterT:
            converter = <Converter*>target
            converter.use(actor, actor_id, &self.env._rewards[actor_id])

        return True
