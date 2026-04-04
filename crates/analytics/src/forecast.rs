use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use tracing::instrument;

use crate::AnalyticsError;

/// Daily review load prediction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DailyLoad {
    pub day: u32,
    pub reviews: usize,
    pub new_learned: usize,
    pub memorized: f32,
}

/// Workload forecast result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkloadForecast {
    pub forecast_days: u32,
    pub total_cards: usize,
    pub total_reviews: usize,
    pub daily: Vec<DailyLoad>,
    pub peak_day: u32,
    pub peak_reviews: usize,
    pub avg_daily_reviews: f32,
}

/// Card scheduling state needed for simulation.
#[derive(sqlx::FromRow)]
struct CardState {
    ivl: i32,
    ease: i32,
    due: Option<i32>,
    lapses: i32,
}

/// Forecast review workload for the next `days_ahead` days.
///
/// Uses the FSRS simulator to predict daily review counts based on
/// current card states (interval, ease, due dates, lapses).
#[instrument(skip(pool))]
pub async fn forecast_workload(
    pool: &PgPool,
    days_ahead: u32,
    deck_filter: Option<&str>,
) -> Result<WorkloadForecast, AnalyticsError> {
    let cards = if let Some(deck) = deck_filter {
        sqlx::query_as::<_, CardState>(
            "SELECT c.ivl, c.ease, c.due, c.lapses
             FROM cards c
             JOIN decks d ON c.deck_id = d.deck_id
             WHERE d.name = $1",
        )
        .bind(deck)
        .fetch_all(pool)
        .await?
    } else {
        sqlx::query_as::<_, CardState>("SELECT ivl, ease, due, lapses FROM cards")
            .fetch_all(pool)
            .await?
    };

    if cards.is_empty() {
        return Ok(WorkloadForecast {
            forecast_days: days_ahead,
            total_cards: 0,
            total_reviews: 0,
            daily: Vec::new(),
            peak_day: 0,
            peak_reviews: 0,
            avg_daily_reviews: 0.0,
        });
    }

    let total_cards = cards.len();

    // Convert to FSRS Card structs via SM-2 parameter bootstrapping
    let fsrs_engine = fsrs::FSRS::new(Some(&[]))
        .map_err(|e| AnalyticsError::Internal(format!("FSRS init: {e}")))?;

    let existing_cards: Vec<fsrs::Card> = cards
        .iter()
        .filter_map(|c| {
            if c.ivl <= 0 {
                return None;
            }
            let ease_factor = c.ease as f32 / 1000.0;
            let interval = c.ivl as f32;
            let state = fsrs_engine
                .memory_state_from_sm2(ease_factor, interval, 0.9)
                .ok()?;

            Some(fsrs::Card {
                id: 0,
                difficulty: state.difficulty,
                stability: state.stability,
                due: c.due.unwrap_or(0) as f32,
                last_date: -(c.ivl as f32),
                interval: c.ivl as f32,
                lapses: c.lapses as u32,
            })
        })
        .collect();

    let config = fsrs::SimulatorConfig {
        deck_size: 0,
        learn_span: days_ahead as usize,
        ..Default::default()
    };

    let result = fsrs::simulate(
        &config,
        &fsrs::DEFAULT_PARAMETERS,
        0.9,
        Some(42),
        Some(existing_cards),
    )
    .map_err(|e| AnalyticsError::Internal(format!("FSRS simulate: {e}")))?;

    let daily: Vec<DailyLoad> = result
        .review_cnt_per_day
        .iter()
        .enumerate()
        .map(|(i, &reviews)| DailyLoad {
            day: i as u32,
            reviews,
            new_learned: result.learn_cnt_per_day.get(i).copied().unwrap_or(0),
            memorized: result.memorized_cnt_per_day.get(i).copied().unwrap_or(0.0),
        })
        .collect();

    let total_reviews: usize = daily.iter().map(|d| d.reviews).sum();
    let (peak_day, peak_reviews) = daily
        .iter()
        .enumerate()
        .max_by_key(|(_, d)| d.reviews)
        .map(|(i, d)| (i as u32, d.reviews))
        .unwrap_or((0, 0));

    let avg = if daily.is_empty() {
        0.0
    } else {
        total_reviews as f32 / daily.len() as f32
    };

    Ok(WorkloadForecast {
        forecast_days: days_ahead,
        total_cards,
        total_reviews,
        daily,
        peak_day,
        peak_reviews,
        avg_daily_reviews: avg,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_forecast_serialization() {
        let forecast = WorkloadForecast {
            forecast_days: 7,
            total_cards: 0,
            total_reviews: 0,
            daily: Vec::new(),
            peak_day: 0,
            peak_reviews: 0,
            avg_daily_reviews: 0.0,
        };
        let json = serde_json::to_string(&forecast).unwrap();
        let restored: WorkloadForecast = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.forecast_days, 7);
        assert_eq!(restored.total_reviews, 0);
    }

    #[test]
    fn daily_load_serialization() {
        let load = DailyLoad {
            day: 1,
            reviews: 25,
            new_learned: 5,
            memorized: 100.0,
        };
        let json = serde_json::to_string(&load).unwrap();
        let restored: DailyLoad = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.reviews, 25);
    }
}
