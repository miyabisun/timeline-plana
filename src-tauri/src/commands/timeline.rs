use crate::core::tactical_guidance::{parse_plan, Guide, Plan};
use crate::state::GuideState;
use std::sync::PoisonError;
use tauri::State;

/// Reads a timeline YAML; battles are guided through it from now on. The errors say
/// what is wrong where.
#[tauri::command]
pub fn load_timeline(yaml: &str, guide: State<'_, GuideState>) -> Result<Plan, Vec<String>> {
    let plan = parse_plan(yaml)?;
    *guide.lock().unwrap_or_else(PoisonError::into_inner) = Some(Guide::new(plan.clone()));
    Ok(plan)
}

/// Goes on from the skill in `row`; false when there is no timeline or no skill there.
#[tauri::command]
pub fn select_timeline_row(row: usize, guide: State<'_, GuideState>) -> bool {
    guide
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .as_mut()
        .is_some_and(|guide| guide.select(row))
}
