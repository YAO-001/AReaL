from types import SimpleNamespace

from areal.engine.awex.sglang_plugin import AwexSchedulerPlugin


def _base_scheduler(**extra):
    fields = dict(
        gpu_id=0,
        event_loop_overlap=lambda: None,
        event_loop_normal=lambda: None,
        recv_requests=lambda: [],
        process_input_requests=lambda reqs: None,
        _engine_paused=False,
        is_generation=True,
        last_batch=None,
        cur_batch=None,
    )
    fields.update(extra)
    return SimpleNamespace(**fields)


def _plugin_for(scheduler):
    plugin = AwexSchedulerPlugin.__new__(AwexSchedulerPlugin)
    plugin._scheduler = scheduler
    plugin._paused_poll_interval_s = 0.001
    return plugin


def test_patch_event_loop_survives_missing_decode_hooks():
    scheduler = _base_scheduler()
    assert not hasattr(scheduler, "log_decode_stats")
    assert not hasattr(scheduler, "log_decode_stats_every_iteration")

    _plugin_for(scheduler)._patch_event_loop()

    assert scheduler.event_loop_overlap.__name__ == "_patched_overlap"
    assert scheduler.event_loop_normal.__name__ == "_patched_normal"
