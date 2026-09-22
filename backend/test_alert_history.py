from datetime import datetime, timedelta, timezone

import app as backend_app
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


def test_safe_prediction_does_not_auto_resolve_active_alert(monkeypatch):
    device_id = 'SENTRY-TEST'
    backend_app.devices[device_id] = backend_app.default_device(device_id)
    backend_app.incidents['incident-test'] = {
        'id': 'incident-test',
        'type': 'EXPLOSIVE PROXY',
        'modelPrediction': 'EXPLOSIVE',
        'location': 'Gate',
        'platform': 'Platform 1',
        'confidence': 92.0,
        'status': 'RESPONSE DISPATCHED',
        'team': 'SECURITY RESPONSE',
        'time': backend_app.utc_now(),
        'device': device_id,
        'predictionId': 'prediction-test',
        'latestReading': None,
    }

    record_alert_event(
        device_id=device_id,
        prediction='EXPLOSIVE',
        display_result='EXPLOSIVE PROXY',
        confidence=0.92,
        location='Gate',
        platform='Platform 1',
        status='THREAT',
    )

    monkeypatch.setattr(
        backend_app,
        'call_ml_server',
        lambda *_args, **_kwargs: {
            'prediction': 'SAFE',
            'confidence': 0.99,
            'probabilities': {'SAFE': 0.99, 'EXPLOSIVE': 0.01},
            'features': {'VMQ2': 1.0, 'VMQ3': 2.0, 'VMQ135': 3.0, 'dVdt_max': 4.0, 'temperature': 25.0, 'humidity': 40.0},
        },
    )

    backend_app.create_prediction(
        device_id,
        [{
            'timestamp': backend_app.utc_now(),
            'temperature': 25.0,
            'humidity': 40.0,
            'mq2': 10.0,
            'mq3': 12.0,
            'mq135': 14.0,
            'source_status': 'ONLINE',
            'test_object': 'metal',
        }],
    )

    assert get_active_alert_event(device_id) is not None
    assert backend_app.incidents['incident-test']['status'] == 'RESPONSE DISPATCHED'


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
