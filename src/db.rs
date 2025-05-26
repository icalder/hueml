use std::{error::Error, fmt, str::FromStr};

use chrono::{DateTime, Utc};
use futures::Stream;
use sqlx::{Pool, Postgres};

/*

/lights/2 => bedside light
/lights/3 => lounge TV light

select id, creationtime, d->'on'->>'on' as state
from v2events as v2, jsonb_array_elements(
    (select data from v2events where id = v2.id)
) as d
where d#>'{on}' is not null and
    d@>'{"id_v1": "/lights/3", "type": "light"}'
order by creationtime
limit 100
 */

#[derive(Debug, PartialEq)]
pub enum LightState {
    On,
    Off,
}

impl fmt::Display for LightState {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl sqlx::Type<Postgres> for LightState {
    fn type_info() -> <Postgres as sqlx::Database>::TypeInfo {
        <String as sqlx::Type<sqlx::Postgres>>::type_info()
    }
}

// DB is the database driver
// `'r` is the lifetime of the `Row` being decoded
impl<'r, DB: sqlx::Database> sqlx::Decode<'r, DB> for LightState
where
    // we want to delegate some of the work to string decoding so let's make sure strings
    // are supported by the database
    &'r str: sqlx::Decode<'r, DB>,
{
    fn decode(
        value: <DB as sqlx::Database>::ValueRef<'r>,
    ) -> Result<LightState, Box<dyn Error + 'static + Send + Sync>> {
        let value = <&str as sqlx::Decode<DB>>::decode(value)?;
        // now you can parse this value into your type (assuming there is a `FromStr`)
        Ok(value.parse()?)
    }
}

impl FromStr for LightState {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "true" => Ok(LightState::On),
            "false" => Ok(LightState::Off),
            _ => Err("unknown light state".to_string()),
        }
    }
}

#[derive(sqlx::FromRow, Debug)]
#[allow(dead_code)]
pub struct LightEvent {
    pub id: String,
    pub creationtime: chrono::NaiveDateTime,
    pub state: LightState,
}

impl fmt::Display for LightEvent {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl LightEvent {
    pub fn utc_datetime(&self) -> DateTime<Utc> {
        DateTime::from_naive_utc_and_offset(self.creationtime, Utc)
    }

    pub fn on(&self) -> bool {
        return self.state == LightState::On;
    }
}

// https://gendignoux.com/blog/2021/04/01/rust-async-streams-futures-part1.html#primer-creating-a-stream-of-pages
pub async fn stream_query<'p>(
    db_pool: &'p Pool<Postgres>,
    sql_buf: &'p mut String,
    from: Option<chrono::NaiveDate>,
    to: Option<chrono::NaiveDate>,
) -> impl Stream<Item = Result<LightEvent, sqlx::Error>> + 'p {
    let mut bind_args = vec![];

    sql_buf.clear();
    sql_buf.push_str(
        r#"select id, creationtime, d->'on'->>'on' as state
from v2events as v2, jsonb_array_elements(
(select data from v2events where id = v2.id)
) as d
where d#>'{on}' is not null and
d@>'{"id_v1": "/lights/3", "type": "light"}'"#,
    );

    if let Some(f) = from {
        sql_buf.push_str(" and creationtime >= $1 ");
        bind_args.push(f);
    }

    if let Some(t) = to {
        if bind_args.len() > 0 {
            sql_buf.push_str(" and creationtime <= $2 ");
        } else {
            sql_buf.push_str(" and creationtime <= $1 ")
        }
        bind_args.push(t);
    }

    sql_buf.push_str("order by creationtime limit 100000");

    let mut q = sqlx::query_as::<_, LightEvent>(sql_buf.as_str());
    for arg in bind_args {
        q = q.bind(arg);
    }

    q.fetch(db_pool)
}
