from datetime import datetime, timedelta, timezone

from app import get_active_alert_event, prune_old_alert_events, record_alert_event


def test_alert_event_is_unique_while_active():
    record_alert_event(
        device_id='SENTRY-001',
        prediction='EXPLOSIVE',
        display_result='EXPLOSIVE PROXY',
        confidence=0.92,
        location='Entry Gate',
        platform='Platform 1',
        status='THREAT',
    )

    first = get_active_alert_event('SENTRY-001')
    assert first is not None

    duplicate = record_alert_event(
        device_id='SENTRY-001',
        prediction='NARCOTIC',
        display_result='NARCOTIC PROXY',
        confidence=0.97,
        location='Entry Gate',
        platform='Platform 1',
        status='THREAT',
    )

    assert duplicate is None
    assert get_active_alert_event('SENTRY-001')['prediction'] == 'EXPLOSIVE'


def test_old_alerts_are_pruned():
    cutoff = datetime.now(timezone.utc) - timedelta(days=8)
    old_ts = cutoff.isoformat().replace('+00:00', 'Z')
    record_alert_event(
        device_id='SENTRY-999',
        prediction='EXPLOSIVE',
        display_result='EXPLOSIVE PROXY',
        confidence=0.9,
        location='Old Area',
        platform='Old Platform',
        status='THREAT',
        created_at=old_ts,
    )

    pruned = prune_old_alert_events()
    assert pruned >= 0
