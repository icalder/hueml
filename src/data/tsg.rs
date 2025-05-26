use std::collections::VecDeque;

use chrono::{DateTime, Duration, DurationRound, Utc};

use crate::db::{LightEvent, LightState};

use super::types::LightSample;

pub struct LightTimeSeriesGenerator {
    sample_interval_mins: u8,
    events: VecDeque<LightEvent>,
    state: bool,
    // The time of the last generated sample (initially zero)
    sample_time: DateTime<Utc>,
    // The event we are currently iterating away from
    //eFrom: Option<LightEvent>,
    // The known event we are currently iterating towards
    next_event: Option<LightEvent>,
    //now: DateTime<Utc>,
    //end: DateTime<Utc>,
}

impl Default for LightTimeSeriesGenerator {
    fn default() -> Self {
        Self {
            sample_interval_mins: 15,
            events: Default::default(),
            state: false,
            sample_time: Default::default(),
            next_event: None,
        }
    }
}

/*
We want to produce output samples every 15 (say) mins at exact 5 min time points, so:
hh:00
hh:05
...
hh:55

Given a single input event, we only know that a light was on or off at time t0 which may
lie in between two five-minute points.  We can't produce any output until we get the *next*
event.  Then we can generate output for all points in between t0 and t1 (exclusive).

Chrono rounding question: https://github.com/chronotope/chrono/issues/280
Duration rounding trait: https://github.com/chronotope/chrono/pull/445

The first output time
 */
impl LightTimeSeriesGenerator {
    pub fn event(&mut self, e: LightEvent) {
        self.events.push_back(e);
    }
}

impl Iterator for LightTimeSeriesGenerator {
    type Item = LightSample;

    fn next(&mut self) -> Option<Self::Item> {
        // Start with happy path.
        // 1. sampleTime is non-zero
        // 2. sampleTime is <= nextEvent - sample interval
        // - this means we can generate a new sample

        // Each time we set a new next_event we must:
        // - set state to the state of the current next_event

        // We can always emit a new sample if sample_time <= max_sample_time
        // where max_sample time is next_event ts - sample_interval

        // What happens the on the first iteration?
        // - we pop the oldest event from events queue - call it e1
        // - we use e1 to set sample_time, by round e1's timestamp to nearest sample_interval minute
        // - from now on sample_time only ever increments by sample_interval.  The timestamp of future
        //   events is ignore except as a bounds check for max sample time
        // - we need to set next_event by pop-ing another event
        // - if there is no event to pop we emit nothing
        // - on the next iteration we may be able to pop a next_event
        // - if next_event was empty but just filled we don't update state, instead we emit current state until we reach max sample time

        let sample_interval_duration = Duration::minutes(self.sample_interval_mins.into());

        if self.sample_time.timestamp() == 0 {
            // This is the first iteration of the generator
            // Initialise the sample time using the first event (if any)
            if let Some(e1) = self.events.pop_front() {
                self.sample_time = e1
                    .utc_datetime()
                    .duration_round(sample_interval_duration)
                    .unwrap();
                self.state = e1.on();
            } else {
                return None;
            }
        }

        // If we don't have a next event then pull one from the front of the queue
        if self.next_event.is_none() {
            self.next_event = self.events.pop_front();
        }

        if self.next_event.is_none() {
            // we need more events
            return None;
        }

        let mut max_sample_time = self.next_event.as_ref().unwrap().utc_datetime();

        while self.sample_time >= max_sample_time {
            // We need to pull another event.  If there are no more events then we can't emit any more samples
            let next_event = self.events.pop_front();
            if next_event.is_none() {
                // we need more events
                return None;
            }
            self.state = self.next_event.as_ref().unwrap().on();
            self.next_event = next_event;
            max_sample_time =
                self.next_event.as_ref().unwrap().utc_datetime() - sample_interval_duration;
        }

        // We can now emit a sample
        let sample_time = self.sample_time;
        self.sample_time += Duration::minutes(self.sample_interval_mins.into());
        Some(LightSample {
            state: if self.state {
                LightState::On
            } else {
                LightState::Off
            },
            time: sample_time,
        })
    }
}
