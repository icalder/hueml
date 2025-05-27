#[cfg(test)]
use super::types::LightSample;
#[cfg(test)]
use crate::data::tsg::LightTimeSeriesGenerator;
#[cfg(test)]
use crate::db::{LightEvent, LightState};
#[cfg(test)]
use chrono::{DateTime, Utc};

// Helper for tests
#[cfg(test)]
fn make_event(id: &str, date_and_time: &str, state: bool) -> LightEvent {
    LightEvent {
        id: String::from(id),
        creationtime: chrono::NaiveDateTime::parse_from_str(date_and_time, "%Y-%m-%d %H:%M:%S")
            .unwrap(),
        state: if state {
            LightState::On
        } else {
            LightState::Off
        },
    }
}

#[cfg(test)]
fn make_lightsample(state: LightState, date_and_time: &str) -> LightSample {
    let t = chrono::NaiveDateTime::parse_from_str(date_and_time, "%Y-%m-%d %H:%M:%S").unwrap();
    LightSample {
        state,
        time: DateTime::from_naive_utc_and_offset(t, Utc),
    }
}

#[test]
fn test_default_datetime() {
    let dt: DateTime<Utc> = DateTime::default();
    // Unix time zero, seconds since 1970
    assert_eq!(0, dt.timestamp());
}

#[test]
fn test_default_light_time_series_generator_is_empty_iterator() {
    let mut tsg = LightTimeSeriesGenerator::default();
    assert_eq!(None, tsg.next());
}

#[test]
fn test_light_time_series_generator_with_single_input_event_produces_no_result() {
    let mut tsg = LightTimeSeriesGenerator::default();
    let e1 = make_event("1", "2023-01-01 16:45:00", true);
    tsg.event(e1);
    assert_eq!(None, tsg.next());
}

#[test]
fn test_light_time_series_generator_with_two_input_events_less_than_one_sample_interval_apart_produces_one_result()
 {
    let mut tsg = LightTimeSeriesGenerator::default().with_sample_interval_mins(5);
    tsg.event(make_event("1", "2023-01-01 16:44:00", true));
    tsg.event(make_event("2", "2023-01-01 16:49:00", false));
    assert_eq!(
        Some(make_lightsample(LightState::On, "2023-01-01 16:45:00")),
        tsg.next()
    );
}

#[test]
fn test_light_time_series_generator_with_two_input_events_one_sample_interval_apart_produces_one_result()
 {
    let mut tsg = LightTimeSeriesGenerator::default().with_sample_interval_mins(5);
    tsg.event(make_event("1", "2023-01-01 16:44:00", true));
    tsg.event(make_event("2", "2023-01-01 16:50:00", false));
    assert_eq!(
        Some(make_lightsample(LightState::On, "2023-01-01 16:45:00")),
        tsg.next()
    );
}

#[test]
fn test_light_time_series_generator_with_two_input_events_just_over_one_sample_interval_apart_produces_two_results()
 {
    let mut tsg = LightTimeSeriesGenerator::default().with_sample_interval_mins(5);
    tsg.event(make_event("1", "2023-01-01 16:44:00", true));
    tsg.event(make_event("2", "2023-01-01 16:50:05", false));
    assert_eq!(
        Some(make_lightsample(LightState::On, "2023-01-01 16:45:00")),
        tsg.next()
    );
    assert_eq!(
        Some(make_lightsample(LightState::On, "2023-01-01 16:50:00")),
        tsg.next()
    );
    assert_eq!(None, tsg.next());
}

#[test]
fn test_light_time_series_generator_with_two_input_events_more_one_sample_interval_apart_produces_multiple_results()
 {
    let mut tsg = LightTimeSeriesGenerator::default().with_sample_interval_mins(5);
    tsg.event(make_event("1", "2023-01-01 16:44:00", true));
    tsg.event(make_event("2", "2023-01-01 17:01:00", false));
    assert_eq!(
        Some(make_lightsample(LightState::On, "2023-01-01 16:45:00")),
        tsg.next()
    );
    assert_eq!(
        Some(make_lightsample(LightState::On, "2023-01-01 16:50:00")),
        tsg.next()
    );
    assert_eq!(
        Some(make_lightsample(LightState::On, "2023-01-01 16:55:00")),
        tsg.next()
    );
    assert_eq!(
        Some(make_lightsample(LightState::On, "2023-01-01 17:00:00")),
        tsg.next()
    );
    assert_eq!(None, tsg.next());
}

#[test]
fn test_light_time_series_generator_resumes_after_waiting_for_a_new_event() {
    let mut tsg = LightTimeSeriesGenerator::default().with_sample_interval_mins(5);
    tsg.event(make_event("1", "2023-01-01 16:44:00", true));
    tsg.event(make_event("2", "2023-01-01 16:50:05", false));
    assert_eq!(
        Some(make_lightsample(LightState::On, "2023-01-01 16:45:00")),
        tsg.next()
    );
    assert_eq!(
        Some(make_lightsample(LightState::On, "2023-01-01 16:50:00")),
        tsg.next()
    );
    assert_eq!(None, tsg.next());
    tsg.event(make_event("3", "2023-01-01 17:15:00", true));
    assert_eq!(
        Some(make_lightsample(LightState::Off, "2023-01-01 16:55:00")),
        tsg.next()
    );
}
