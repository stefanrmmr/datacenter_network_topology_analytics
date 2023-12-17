select fk_simulation_id, simulation.fk_topology_id, max(completion_time) as max from flowcompletiontime
INNER JOIN simulation ON flowcompletiontime.fk_simulation_id=simulation.simulation_id;

SELECT simulation_id, fk_topology_id, max_time FROM simulation AS s
                  INNER JOIN (SELECT fk_simulation_id, max(completion_time) as max_time FROM flowcompletiontime
                      group by fk_simulation_id) AS f ON f.fk_simulation_id=s.simulation_id;


