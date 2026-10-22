//! Tactical Guidance (戦術指導)
//!
//! Reads a timeline YAML made in Timeline Arona and, from the remaining time, the EX
//! cost and the battle state, tells the Sensei which EX skill to use next and when.
//!
//! A timeline is a list of rows read in order. A skill row with `when_at` (remaining
//! time) or `when_cost` (EX cost) opens a group; the skill rows after it without a
//! trigger follow it at once. A `time` row is a marker: shown, never waited for.
//!
//! Plana cannot see which card the Sensei pressed, so a skill counts as used when
//! the EX cost drops between two frames of the running battle.

use crate::core::combat_intel::BattleState;
use serde::{Deserialize, Serialize};

/// A drop of the EX cost this large between two running frames means a skill was used.
const USED_COST_DROP: f32 = 0.9;
/// The remaining time going up by this much means a new battle started.
const NEW_BATTLE_MS: u32 = 2000;
/// How far below a cost threshold a reading may be and still reach it: the reading is
/// an f32, so 7.1 reads as 7.0999999.
const COST_TOLERANCE: f64 = 0.001;
/// A marker this close is about to come.
const SOON_MS: u32 = 10_000;

/// What makes a group's first skill due.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Trigger {
    /// When the remaining time reaches `ms` (as written: `text`).
    At { ms: u32, text: String },
    /// When the EX cost reaches `cost` (f64, so 7.1 reaches the frontend as 7.1).
    Cost { cost: f64 },
}

/// One row of the timeline.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Row {
    /// Use `skill`'s EX: when `trigger` says, or right after the previous skill.
    Skill {
        trigger: Option<Trigger>,
        skill: String,
        target: Option<String>,
        note: Option<String>,
    },
    /// Something to know at a remaining time; nothing to press.
    Mark { ms: u32, time: String, note: String },
}

/// A timeline as Plana uses it.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Plan {
    pub boss: String,
    pub difficulty: Option<String>,
    pub students: Vec<String>,
    pub rows: Vec<Row>,
}

#[derive(Deserialize)]
struct RawPlan {
    boss: Option<String>,
    difficulty: Option<String>,
    party: Option<RawParty>,
    timeline: Option<Vec<RawRow>>,
}

#[derive(Deserialize)]
struct RawParty {
    #[serde(default)]
    striker: Vec<RawStudent>,
    #[serde(default)]
    special: Vec<RawStudent>,
}

#[derive(Deserialize)]
struct RawStudent {
    name: Option<String>,
}

#[derive(Deserialize)]
struct RawRow {
    when_at: Option<String>,
    when_cost: Option<f64>,
    skill: Option<String>,
    target: Option<String>,
    note: Option<String>,
    time: Option<String>,
}

/// Remaining time written `M:SS` or `M:SS.mmm`, in milliseconds.
pub fn parse_time(text: &str) -> Option<u32> {
    let (minutes, rest) = text.split_once(':')?;
    let (seconds, millis) = match rest.split_once('.') {
        Some((s, ms)) if ms.len() == 3 => (s, ms),
        Some(_) => return None,
        None => (rest, "000"),
    };
    let all_digits = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
    if !all_digits(minutes) || seconds.len() != 2 || !all_digits(seconds) || !all_digits(millis) {
        return None;
    }
    let minutes: u32 = minutes.parse().ok()?;
    let seconds: u32 = seconds.parse().ok()?;
    let millis: u32 = millis.parse().ok()?;
    (seconds < 60).then(|| (minutes * 60 + seconds) * 1000 + millis)
}

/// Reads a timeline. Every problem found is listed, each saying where it is.
pub fn parse_plan(yaml: &str) -> Result<Plan, Vec<String>> {
    let raw: RawPlan = serde_norway::from_str(yaml).map_err(|e| vec![format!("YAML: {e}")])?;
    let mut errors = Vec::new();
    let boss = raw.boss.unwrap_or_else(|| {
        errors.push("boss がありません".to_owned());
        String::new()
    });
    let mut students = Vec::new();
    match raw.party {
        Some(party) => {
            for (group, list) in [("striker", party.striker), ("special", party.special)] {
                for (i, student) in list.into_iter().enumerate() {
                    match student.name {
                        Some(name) => students.push(name),
                        None => errors.push(format!(
                            "party.{group} の {} 人目: name がありません",
                            i + 1
                        )),
                    }
                }
            }
        }
        None => errors.push("party がありません".to_owned()),
    }
    let mut rows = Vec::new();
    match raw.timeline {
        Some(timeline) => {
            for (i, raw_row) in timeline.into_iter().enumerate() {
                match read_row(raw_row, &students) {
                    Ok(row) => rows.push(row),
                    Err(problem) => errors.push(format!("timeline の {} 行目: {problem}", i + 1)),
                }
            }
        }
        None => errors.push("timeline がありません".to_owned()),
    }
    if errors.is_empty() {
        Ok(Plan {
            boss,
            difficulty: raw.difficulty,
            students,
            rows,
        })
    } else {
        Err(errors)
    }
}

fn read_row(raw: RawRow, students: &[String]) -> Result<Row, String> {
    if let Some(time) = raw.time {
        if raw.skill.is_some() || raw.when_at.is_some() || raw.when_cost.is_some() {
            return Err("time の行に skill・when_at・when_cost は書けません".to_owned());
        }
        let ms = parse_time(&time)
            .ok_or_else(|| format!("time \"{time}\" は M:SS か M:SS.mmm で書きます"))?;
        let note = raw.note.ok_or("time の行には note が要ります")?;
        return Ok(Row::Mark { ms, time, note });
    }
    let skill = raw.skill.ok_or("skill か time が要ります")?;
    if !students.contains(&skill) {
        return Err(format!("skill \"{skill}\" は party にいません"));
    }
    let trigger = match (raw.when_at, raw.when_cost) {
        (Some(_), Some(_)) => return Err("when_at と when_cost はどちらか一つです".to_owned()),
        (Some(text), None) => {
            let ms = parse_time(&text)
                .ok_or_else(|| format!("when_at \"{text}\" は M:SS か M:SS.mmm で書きます"))?;
            Some(Trigger::At { ms, text })
        }
        (None, Some(cost)) => {
            if !(0.0..=10.0).contains(&cost) {
                return Err(format!("when_cost {cost} は 0〜10 で書きます"));
            }
            Some(Trigger::Cost { cost })
        }
        (None, None) => None,
    };
    Ok(Row::Skill {
        trigger,
        skill,
        target: raw.target,
        note: raw.note,
    })
}

/// The game as Plana read it in one frame.
#[derive(Debug, Clone, Copy)]
pub struct Observation {
    pub time_ms: Option<u32>,
    pub cost: Option<f32>,
    pub state: BattleState,
}

/// One skill of the group shown.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct GuideSkill {
    pub row: usize,
    pub skill: String,
    pub target: Option<String>,
    pub note: Option<String>,
    pub done: bool,
}

/// A marker still to come.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct GuideMark {
    pub row: usize,
    pub ms: u32,
    pub time: String,
    pub note: String,
    /// Whether it comes within the next 10 seconds.
    pub soon: bool,
}

/// What to show the Sensei now.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Guidance {
    /// The row of the next skill to use; `None` once every skill is used.
    pub next: Option<usize>,
    /// The trigger of the group the next skill is in (`None`: no trigger, use it now).
    pub trigger: Option<Trigger>,
    /// Whether the next skill is due.
    pub due: bool,
    /// The skills of that group, those already used marked `done`.
    pub skills: Vec<GuideSkill>,
    /// The next marker to come, by the remaining time.
    pub mark: Option<GuideMark>,
}

/// Follows a battle through a plan.
#[derive(Debug, Clone)]
pub struct Guide {
    plan: Plan,
    /// The row of the next skill to use.
    next: Option<usize>,
    /// The EX cost in the last frame of the running battle.
    last_cost: Option<f32>,
    /// The remaining time last read.
    last_time_ms: Option<u32>,
}

impl Guide {
    pub fn new(plan: Plan) -> Self {
        let next = first_skill_from(&plan.rows, 0);
        Self {
            plan,
            next,
            last_cost: None,
            last_time_ms: None,
        }
    }

    pub fn plan(&self) -> &Plan {
        &self.plan
    }

    /// Continues from `row` (the Sensei's choice after a misread). False when it is
    /// not a skill row.
    pub fn select(&mut self, row: usize) -> bool {
        let is_skill = matches!(self.plan.rows.get(row), Some(Row::Skill { .. }));
        if is_skill {
            self.next = Some(row);
        }
        is_skill
    }

    /// Takes in one frame: a new battle starts the plan over, a used skill moves on.
    pub fn observe(&mut self, frame: &Observation) {
        if let Some(time) = frame.time_ms {
            if self
                .last_time_ms
                .is_some_and(|last| time > last + NEW_BATTLE_MS)
            {
                self.next = first_skill_from(&self.plan.rows, 0);
                self.last_cost = None;
            }
            self.last_time_ms = Some(time);
        }
        if frame.state != BattleState::Active {
            return;
        }
        let Some(cost) = frame.cost else {
            return;
        };
        if let Some(next) = self.next {
            if self
                .last_cost
                .is_some_and(|last| last - cost >= USED_COST_DROP)
            {
                self.next = first_skill_from(&self.plan.rows, next + 1);
            }
        }
        self.last_cost = Some(cost);
    }

    /// What to show for `frame`.
    pub fn guidance(&self, frame: &Observation) -> Guidance {
        let rows = &self.plan.rows;
        let mark = self.next_mark(frame.time_ms);
        let Some(next) = self.next else {
            return Guidance {
                next: None,
                trigger: None,
                due: false,
                skills: Vec::new(),
                mark,
            };
        };
        let start = (0..=next)
            .rev()
            .find(|&i| {
                matches!(
                    rows[i],
                    Row::Skill {
                        trigger: Some(_),
                        ..
                    }
                )
            })
            .unwrap_or(0);
        let trigger = match &rows[start] {
            Row::Skill { trigger, .. } => trigger.clone(),
            Row::Mark { .. } => None,
        };
        let skills = rows
            .iter()
            .enumerate()
            .skip(start)
            .take_while(|&(i, row)| {
                i == start
                    || !matches!(
                        row,
                        Row::Skill {
                            trigger: Some(_),
                            ..
                        }
                    )
            })
            .filter_map(|(i, row)| match row {
                Row::Skill {
                    skill,
                    target,
                    note,
                    ..
                } => Some(GuideSkill {
                    row: i,
                    skill: skill.clone(),
                    target: target.clone(),
                    note: note.clone(),
                    done: i < next,
                }),
                Row::Mark { .. } => None,
            })
            .collect();
        let due = next != start
            || match &trigger {
                None => true,
                Some(Trigger::At { ms, .. }) => frame.time_ms.is_some_and(|t| t <= *ms),
                Some(Trigger::Cost { cost }) => frame
                    .cost
                    .is_some_and(|c| f64::from(c) + COST_TOLERANCE >= *cost),
            };
        Guidance {
            next: Some(next),
            trigger,
            due,
            skills,
            mark,
        }
    }

    /// The marker that comes next as the time runs down: the latest one not yet passed.
    fn next_mark(&self, time_ms: Option<u32>) -> Option<GuideMark> {
        let time = time_ms?;
        self.plan
            .rows
            .iter()
            .enumerate()
            .filter_map(|(i, row)| match row {
                Row::Mark {
                    ms,
                    time: text,
                    note,
                } if *ms <= time => Some((*ms, i, text, note)),
                _ => None,
            })
            .max_by_key(|&(ms, i, _, _)| (ms, std::cmp::Reverse(i)))
            .map(|(ms, row, text, note)| GuideMark {
                row,
                ms,
                time: text.clone(),
                note: note.clone(),
                soon: time - ms <= SOON_MS,
            })
    }
}

fn first_skill_from(rows: &[Row], from: usize) -> Option<usize> {
    (from..rows.len()).find(|&i| matches!(rows[i], Row::Skill { .. }))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The example of timeline-arona's docs/README.md.
    const EXAMPLE: &str = include_str!("../../../e2e/fixtures/example.yaml");

    fn example() -> Guide {
        Guide::new(parse_plan(EXAMPLE).unwrap())
    }

    fn frame(time: &str, cost: f32, state: BattleState) -> Observation {
        Observation {
            time_ms: parse_time(time),
            cost: Some(cost),
            state,
        }
    }

    fn running(time: &str, cost: f32) -> Observation {
        frame(time, cost, BattleState::Active)
    }

    fn names(guidance: &Guidance) -> Vec<(&str, bool)> {
        guidance
            .skills
            .iter()
            .map(|s| (s.skill.as_str(), s.done))
            .collect()
    }

    #[test]
    fn reads_remaining_times_as_written_in_the_docs() {
        assert_eq!(parse_time("3:30"), Some(210_000));
        assert_eq!(parse_time("1:20.500"), Some(80_500));
        assert_eq!(parse_time("0:05"), Some(5_000));
        for bad in ["3:3", "3:60", "3:30.5", "a:30", "330", ":30", "3:30."] {
            assert_eq!(parse_time(bad), None, "{bad}");
        }
    }

    #[test]
    fn reads_the_docs_example() {
        let plan = parse_plan(EXAMPLE).unwrap();
        assert_eq!(plan.boss, "ホド");
        assert_eq!(plan.students, ["シロコ", "ホシノ", "コタマ", "セリナ"]);
        assert_eq!(plan.rows.len(), 9);
        assert_eq!(
            plan.rows[0],
            Row::Skill {
                trigger: Some(Trigger::At {
                    ms: 210_000,
                    text: "3:30".to_owned()
                }),
                skill: "コタマ".to_owned(),
                target: None,
                note: Some("開幕バフ".to_owned()),
            }
        );
        assert!(
            matches!(&plan.rows[1], Row::Skill { trigger: None, skill, .. } if skill == "ホシノ")
        );
        assert!(matches!(&plan.rows[3], Row::Mark { ms: 205_000, .. }));
        assert!(matches!(
            &plan.rows[4],
            Row::Skill { trigger: Some(Trigger::Cost { cost }), .. } if (*cost - 7.1).abs() < 1e-6
        ));
        assert!(matches!(&plan.rows[8], Row::Skill { target: Some(t), .. } if t == "シロコ"));
    }

    #[test]
    fn lists_every_problem_with_where_it_is() {
        let yaml = r#"
party:
  striker:
    - name: "シロコ"
timeline:
  - when_at: "3:3"
    skill: "シロコ"
  - skill: "ヒナ"
  - when_at: "3:00"
    when_cost: 3
    skill: "シロコ"
  - time: "2:00"
  - when_cost: 12
    skill: "シロコ"
"#;
        let errors = parse_plan(yaml).unwrap_err();
        assert_eq!(
            errors,
            [
                "boss がありません",
                "timeline の 1 行目: when_at \"3:3\" は M:SS か M:SS.mmm で書きます",
                "timeline の 2 行目: skill \"ヒナ\" は party にいません",
                "timeline の 3 行目: when_at と when_cost はどちらか一つです",
                "timeline の 4 行目: time の行には note が要ります",
                "timeline の 5 行目: when_cost 12 は 0〜10 で書きます",
            ]
        );
    }

    #[test]
    fn says_why_a_file_is_no_timeline() {
        let errors = parse_plan("boss: [").unwrap_err();
        assert_eq!(errors.len(), 1, "{errors:?}");
        assert!(errors[0].starts_with("YAML: "), "{errors:?}");
        assert_eq!(
            parse_plan("boss: \"ホド\"").unwrap_err(),
            ["party がありません", "timeline がありません"]
        );
    }

    #[test]
    fn the_first_group_waits_for_its_time() {
        let guide = example();
        let before = guide.guidance(&running("3:40", 3.0));
        assert_eq!(before.next, Some(0));
        assert!(!before.due, "3:40 is before 3:30");
        assert_eq!(
            names(&before),
            [("コタマ", false), ("ホシノ", false), ("シロコ", false)]
        );
        assert!(
            guide.guidance(&running("3:30", 3.0)).due,
            "3:30 is the time"
        );
        assert!(
            guide.guidance(&running("3:29.967", 3.0)).due,
            "and after it"
        );
        let unread = Observation {
            time_ms: None,
            cost: Some(3.0),
            state: BattleState::Active,
        };
        assert!(
            !guide.guidance(&unread).due,
            "an unread timer is not the time"
        );
    }

    #[test]
    fn a_used_skill_moves_on_to_the_one_that_follows_at_once() {
        let mut guide = example();
        guide.observe(&running("3:31", 5.0));
        guide.observe(&running("3:30", 5.1));
        guide.observe(&running("3:29.900", 3.1));
        let now = guide.guidance(&running("3:29.900", 3.1));
        assert_eq!(now.next, Some(1));
        assert!(now.due, "ホシノ follows コタマ at once");
        assert_eq!(
            names(&now),
            [("コタマ", true), ("ホシノ", false), ("シロコ", false)]
        );
    }

    #[test]
    fn regained_cost_and_small_misreads_use_nothing() {
        let mut guide = example();
        for (time, cost) in [("3:31", 5.0), ("3:30", 5.5), ("3:29", 4.8), ("3:28", 6.0)] {
            guide.observe(&running(time, cost));
        }
        assert_eq!(guide.guidance(&running("3:28", 6.0)).next, Some(0));
    }

    #[test]
    fn a_skill_used_in_slow_motion_counts_when_the_battle_runs_again() {
        let mut guide = example();
        guide.observe(&running("3:30", 5.0));
        guide.observe(&frame("3:30", 1.0, BattleState::Slow));
        assert_eq!(
            guide.guidance(&running("3:30", 1.0)).next,
            Some(0),
            "not while slow"
        );
        guide.observe(&frame("3:29", 3.0, BattleState::Paused));
        guide.observe(&running("3:29", 3.0));
        assert_eq!(guide.guidance(&running("3:29", 3.0)).next, Some(1));
    }

    #[test]
    fn markers_are_shown_but_never_waited_for() {
        let mut guide = example();
        guide.observe(&running("3:30", 9.0));
        for cost in [7.0, 5.0, 3.0] {
            guide.observe(&running("3:29", cost));
        }
        let now = guide.guidance(&running("3:28", 3.0));
        assert_eq!(now.next, Some(4), "the marker row 3 is skipped");
        assert_eq!(
            names(&now),
            [("コタマ", false), ("ホシノ", false), ("シロコ", false)]
        );
        assert!(matches!(now.trigger, Some(Trigger::Cost { .. })));
        assert!(!now.due, "cost 3.0 is under 7.1");
        assert!(guide.guidance(&running("3:20", 7.1)).due);
        let mark = now.mark.unwrap();
        assert_eq!((mark.row, mark.time.as_str()), (3, "3:25"));
        assert!(mark.soon, "3:25 is 3 seconds away");
        let far = guide.guidance(&running("3:40", 3.0)).mark.unwrap();
        assert!(!far.soon, "3:25 is 15 seconds away from 3:40");
        assert!(
            guide.guidance(&running("3:35", 3.0)).mark.unwrap().soon,
            "and 10 seconds away is soon"
        );
        let later = guide.guidance(&running("3:24", 3.0)).mark.unwrap();
        assert_eq!(later.time, "1:30", "3:25 has passed");
        assert_eq!(guide.guidance(&running("1:29", 3.0)).mark, None);
    }

    #[test]
    fn after_the_last_skill_nothing_is_next() {
        let mut guide = example();
        assert!(guide.select(8));
        guide.observe(&running("1:20", 5.0));
        guide.observe(&running("1:19", 2.0));
        let done = guide.guidance(&running("1:19", 2.0));
        assert_eq!(done.next, None);
        assert!(done.skills.is_empty());
    }

    #[test]
    fn a_new_battle_starts_the_plan_over() {
        let mut guide = example();
        assert!(guide.select(5));
        guide.observe(&running("0:10", 5.0));
        guide.observe(&running("3:59", 3.0));
        let fresh = guide.guidance(&running("3:59", 3.0));
        assert_eq!(
            fresh.next,
            Some(0),
            "the timer went up: a new battle, and the old battle's cost 5.0 is no drop to 3.0"
        );
    }

    /// The plan and the guidance as the frontend receives them; the e2e tests send
    /// these same files, so the two sides cannot drift apart.
    #[test]
    fn the_frontend_gets_the_shapes_the_e2e_tests_use() {
        let plan = parse_plan(EXAMPLE).unwrap();
        let mut guide = Guide::new(plan.clone());
        for (time, cost) in [("3:31", 5.0), ("3:30", 5.1), ("3:29.900", 3.1)] {
            guide.observe(&running(time, cost));
        }
        let guidance = guide.guidance(&running("3:29.900", 3.1));
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../e2e/fixtures/");
        for (name, value) in [
            ("example-plan.json", serde_json::to_value(&plan).unwrap()),
            (
                "example-guidance.json",
                serde_json::to_value(&guidance).unwrap(),
            ),
        ] {
            let path = format!("{dir}{name}");
            let written = std::fs::read_to_string(&path).unwrap_or_default();
            let expected = serde_json::to_string_pretty(&value).unwrap() + "\n";
            if std::env::var_os("UPDATE_FIXTURES").is_some() {
                std::fs::write(&path, &expected).unwrap();
            } else {
                assert_eq!(
                    written.replace("\r\n", "\n"),
                    expected,
                    "{name} is stale: run with UPDATE_FIXTURES=1"
                );
            }
        }
    }

    #[test]
    fn the_sensei_picks_the_skill_to_go_on_from() {
        let mut guide = example();
        assert!(!guide.select(3), "a marker is no skill");
        assert!(!guide.select(99));
        assert!(guide.select(6));
        let now = guide.guidance(&running("2:00", 3.0));
        assert_eq!(now.next, Some(6));
        assert!(
            now.due,
            "in a group, the skill after a used one is due, though cost 3.0 is under the group's 7.1"
        );
        assert_eq!(
            names(&now),
            [("コタマ", true), ("ホシノ", true), ("シロコ", false)]
        );
    }
}
