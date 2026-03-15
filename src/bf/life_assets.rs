//! Repository-local physical Conway's Life component assets.
//!
//! Parsing a manifest is deliberately separate from verification. Only
//! [`VerifiedAssetRegistry`] exposes component lookup, and it can be created only
//! after every v1 component and all of its integrity metadata have been checked.

use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;
use std::fs;
use std::path::{Component, Path, PathBuf};

const MANIFEST_SCHEMA: &str = "asimplelife/bf-life-assets/v1";
const REPOSITORY_MANIFEST: &str = "assets/bf_life/manifest.json";
const MAX_ASSET_BYTES: u64 = 64 * 1024 * 1024;
const MAX_PATTERN_CELLS: u64 = 8 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComponentKind {
    Clock,
    Conduit,
    Delay,
    Splitter,
    Crossover,
    Eater,
    StateLatch,
    HeadTokenMover,
    BitIncrement,
    BitDecrement,
    ZeroDetector,
    ConditionalRouter,
    HaltLatch,
    OutputBitTransducer,
}

pub const REQUIRED_V1_COMPONENT_KINDS: &[ComponentKind] = &[
    ComponentKind::Clock,
    ComponentKind::Conduit,
    ComponentKind::Delay,
    ComponentKind::Splitter,
    ComponentKind::Crossover,
    ComponentKind::Eater,
    ComponentKind::StateLatch,
    ComponentKind::HeadTokenMover,
    ComponentKind::BitIncrement,
    ComponentKind::BitDecrement,
    ComponentKind::ZeroDetector,
    ComponentKind::ConditionalRouter,
    ComponentKind::HaltLatch,
    ComponentKind::OutputBitTransducer,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ManifestStatus {
    Blocked,
    Verified,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AssetManifest {
    schema: String,
    status: ManifestStatus,
    pub components: Vec<AssetComponent>,
    pub blocker: Option<AssetBlocker>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AssetBlocker {
    pub reason: String,
    pub missing_components: Vec<ComponentKind>,
    pub required_tape_cells: u8,
    pub required_cell_bits: u8,
    pub input_supported: bool,
    pub output_encoding: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AssetComponent {
    pub id: String,
    pub kind: ComponentKind,
    pub pattern: AssetPattern,
    pub provenance: Provenance,
    pub bounds: AssetBounds,
    pub ports: Vec<AssetPort>,
    pub period: u64,
    pub phase: u64,
    pub isolation: Isolation,
    pub verification: VerificationRecord,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PatternFormat {
    Rle,
    Coordinates,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AssetPattern {
    pub path: String,
    pub format: PatternFormat,
    pub sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Provenance {
    pub author: String,
    pub source_url: String,
    pub source_commit: String,
    pub license: String,
    pub license_url: String,
    pub redistribution_permitted: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AssetBounds {
    pub min_x: i64,
    pub min_y: i64,
    pub max_x: i64,
    pub max_y: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PortRole {
    Input,
    Output,
    Bidirectional,
    Observation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PortDirection {
    North,
    NorthEast,
    East,
    SouthEast,
    South,
    SouthWest,
    West,
    NorthWest,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AssetPort {
    pub name: String,
    pub x: i64,
    pub y: i64,
    pub role: PortRole,
    pub direction: PortDirection,
    pub phase: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Isolation {
    pub left: u64,
    pub right: u64,
    pub top: u64,
    pub bottom: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VerificationRecord {
    pub independently_verified: bool,
    pub verifier: String,
    pub method: String,
    pub evidence: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LifeAssetError {
    Io {
        path: PathBuf,
        message: String,
    },
    ManifestSyntax(String),
    UnsupportedSchema(String),
    ManifestBlocked {
        reason: String,
    },
    UnsafePath(String),
    DuplicateId(String),
    DuplicateKind(ComponentKind),
    MissingRequiredComponents(Vec<ComponentKind>),
    InvalidProvenance {
        id: String,
        field: &'static str,
    },
    RedistributionForbidden(String),
    InvalidSha256 {
        id: String,
        value: String,
    },
    DigestMismatch {
        id: String,
        expected: String,
        actual: String,
    },
    PatternSyntax {
        id: String,
        message: String,
    },
    PatternTooLarge {
        id: String,
        limit: u64,
    },
    EmptyPattern(String),
    InvalidBounds(String),
    CellOutsideBounds {
        id: String,
        cell: (i64, i64),
    },
    InvalidPort {
        id: String,
        message: String,
    },
    InvalidTiming(String),
    InvalidIsolation(String),
    NotIndependentlyVerified(String),
    InvalidVerification {
        id: String,
        field: &'static str,
    },
}

impl fmt::Display for LifeAssetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, message } => write!(f, "{}: {message}", path.display()),
            Self::ManifestSyntax(message) => write!(f, "invalid BF Life manifest: {message}"),
            Self::UnsupportedSchema(schema) => {
                write!(f, "unsupported BF Life asset schema {schema:?}")
            }
            Self::ManifestBlocked { reason } => {
                write!(f, "BF Life asset manifest is blocked: {reason}")
            }
            Self::UnsafePath(path) => write!(f, "asset path is not repository-local: {path:?}"),
            Self::DuplicateId(id) => write!(f, "duplicate BF Life asset id {id:?}"),
            Self::DuplicateKind(kind) => write!(f, "duplicate BF Life v1 component kind {kind:?}"),
            Self::MissingRequiredComponents(kinds) => {
                write!(f, "missing required BF Life v1 components: {kinds:?}")
            }
            Self::InvalidProvenance { id, field } => {
                write!(f, "asset {id:?} has invalid provenance field {field}")
            }
            Self::RedistributionForbidden(id) => {
                write!(f, "asset {id:?} does not permit redistribution")
            }
            Self::InvalidSha256 { id, value } => {
                write!(f, "asset {id:?} has invalid SHA-256 {value:?}")
            }
            Self::DigestMismatch {
                id,
                expected,
                actual,
            } => write!(
                f,
                "asset {id:?} SHA-256 mismatch: expected {expected}, got {actual}"
            ),
            Self::PatternSyntax { id, message } => {
                write!(f, "asset {id:?} pattern is invalid: {message}")
            }
            Self::PatternTooLarge { id, limit } => {
                write!(f, "asset {id:?} exceeds its verified size limit of {limit}")
            }
            Self::EmptyPattern(id) => write!(f, "asset {id:?} pattern contains no live cells"),
            Self::InvalidBounds(id) => write!(f, "asset {id:?} has invalid bounds"),
            Self::CellOutsideBounds { id, cell } => {
                write!(f, "asset {id:?} cell {cell:?} is outside declared bounds")
            }
            Self::InvalidPort { id, message } => {
                write!(f, "asset {id:?} port metadata is invalid: {message}")
            }
            Self::InvalidTiming(id) => write!(f, "asset {id:?} has invalid period or phase"),
            Self::InvalidIsolation(id) => write!(f, "asset {id:?} has invalid isolation clearance"),
            Self::NotIndependentlyVerified(id) => {
                write!(f, "asset {id:?} is not independently verified")
            }
            Self::InvalidVerification { id, field } => {
                write!(f, "asset {id:?} has invalid verification field {field}")
            }
        }
    }
}

impl Error for LifeAssetError {}

#[derive(Debug, Clone)]
pub struct AssetRegistry {
    root: PathBuf,
    manifest: AssetManifest,
}

#[derive(Debug, Clone)]
pub struct VerifiedAssetRegistry {
    components: BTreeMap<ComponentKind, AssetComponent>,
    cells: BTreeMap<ComponentKind, Vec<(i64, i64)>>,
}

impl AssetRegistry {
    pub fn load_repository() -> Result<Self, LifeAssetError> {
        Self::load(Path::new(env!("CARGO_MANIFEST_DIR")).join(REPOSITORY_MANIFEST))
    }

    pub fn load(manifest_path: impl AsRef<Path>) -> Result<Self, LifeAssetError> {
        let manifest_path = manifest_path.as_ref();
        let bytes = fs::read(manifest_path).map_err(|error| LifeAssetError::Io {
            path: manifest_path.to_path_buf(),
            message: error.to_string(),
        })?;
        let manifest = serde_json::from_slice(&bytes)
            .map_err(|error| LifeAssetError::ManifestSyntax(error.to_string()))?;
        let root = manifest_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf();
        Ok(Self { root, manifest })
    }

    pub fn manifest(&self) -> &AssetManifest {
        &self.manifest
    }

    pub fn verify(self) -> Result<VerifiedAssetRegistry, LifeAssetError> {
        if self.manifest.schema != MANIFEST_SCHEMA {
            return Err(LifeAssetError::UnsupportedSchema(self.manifest.schema));
        }
        if self.manifest.status != ManifestStatus::Verified {
            let blocker = self.manifest.blocker.as_ref().ok_or_else(|| {
                LifeAssetError::ManifestSyntax(
                    "blocked manifest must state a blocker reason".to_string(),
                )
            })?;
            validate_blocker(blocker)?;
            return Err(LifeAssetError::ManifestBlocked {
                reason: blocker.reason.trim().to_string(),
            });
        }
        if self.manifest.blocker.is_some() {
            return Err(LifeAssetError::ManifestSyntax(
                "verified manifest cannot retain a blocker record".to_string(),
            ));
        }

        let mut ids = BTreeSet::new();
        let mut components = BTreeMap::new();
        let mut cells = BTreeMap::new();
        for component in self.manifest.components {
            if !ids.insert(component.id.clone()) {
                return Err(LifeAssetError::DuplicateId(component.id));
            }
            if components.contains_key(&component.kind) {
                return Err(LifeAssetError::DuplicateKind(component.kind));
            }
            let parsed = verify_component(&self.root, &component)?;
            cells.insert(component.kind, parsed);
            components.insert(component.kind, component);
        }

        let missing = REQUIRED_V1_COMPONENT_KINDS
            .iter()
            .copied()
            .filter(|kind| !components.contains_key(kind))
            .collect::<Vec<_>>();
        if !missing.is_empty() {
            return Err(LifeAssetError::MissingRequiredComponents(missing));
        }
        Ok(VerifiedAssetRegistry { components, cells })
    }
}

fn validate_blocker(blocker: &AssetBlocker) -> Result<(), LifeAssetError> {
    let missing = blocker
        .missing_components
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let required = REQUIRED_V1_COMPONENT_KINDS
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    if blocker.reason.trim().is_empty()
        || missing != required
        || blocker.missing_components.len() != REQUIRED_V1_COMPONENT_KINDS.len()
        || blocker.required_tape_cells != 64
        || blocker.required_cell_bits != 8
        || blocker.input_supported
        || blocker.output_encoding.trim().is_empty()
    {
        return Err(LifeAssetError::ManifestSyntax(
            "blocked manifest does not match the v1 physical machine contract".to_string(),
        ));
    }
    Ok(())
}

impl VerifiedAssetRegistry {
    pub fn component(&self, kind: ComponentKind) -> &AssetComponent {
        // Construction proves that every required v1 kind is present.
        &self.components[&kind]
    }

    pub fn cells(&self, kind: ComponentKind) -> &[(i64, i64)] {
        &self.cells[&kind]
    }

    pub fn len(&self) -> usize {
        self.components.len()
    }

    pub fn is_empty(&self) -> bool {
        self.components.is_empty()
    }
}

fn verify_component(
    root: &Path,
    component: &AssetComponent,
) -> Result<Vec<(i64, i64)>, LifeAssetError> {
    verify_provenance(component)?;
    verify_metadata(component)?;
    let relative = safe_relative_path(&component.pattern.path)?;
    let canonical_root = fs::canonicalize(root).map_err(|error| LifeAssetError::Io {
        path: root.to_path_buf(),
        message: error.to_string(),
    })?;
    let requested_path = root.join(relative);
    let path = fs::canonicalize(&requested_path).map_err(|error| LifeAssetError::Io {
        path: requested_path,
        message: error.to_string(),
    })?;
    if !path.starts_with(&canonical_root) {
        return Err(LifeAssetError::UnsafePath(component.pattern.path.clone()));
    }
    let asset_len = fs::metadata(&path)
        .map_err(|error| LifeAssetError::Io {
            path: path.clone(),
            message: error.to_string(),
        })?
        .len();
    if asset_len > MAX_ASSET_BYTES {
        return Err(LifeAssetError::PatternTooLarge {
            id: component.id.clone(),
            limit: MAX_ASSET_BYTES,
        });
    }
    let bytes = fs::read(&path).map_err(|error| LifeAssetError::Io {
        path: path.clone(),
        message: error.to_string(),
    })?;
    let expected = component.pattern.sha256.to_ascii_lowercase();
    if expected.len() != 64 || !expected.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(LifeAssetError::InvalidSha256 {
            id: component.id.clone(),
            value: component.pattern.sha256.clone(),
        });
    }
    let actual = sha256_hex(&bytes);
    if actual != expected {
        return Err(LifeAssetError::DigestMismatch {
            id: component.id.clone(),
            expected,
            actual,
        });
    }
    let text = std::str::from_utf8(&bytes).map_err(|error| LifeAssetError::PatternSyntax {
        id: component.id.clone(),
        message: format!("pattern is not UTF-8: {error}"),
    })?;
    let parsed = parse_pattern(text, component.pattern.format).map_err(|message| {
        LifeAssetError::PatternSyntax {
            id: component.id.clone(),
            message,
        }
    })?;
    if parsed.is_empty() {
        return Err(LifeAssetError::EmptyPattern(component.id.clone()));
    }
    for &cell in &parsed {
        if !component.bounds.contains(cell) {
            return Err(LifeAssetError::CellOutsideBounds {
                id: component.id.clone(),
                cell,
            });
        }
    }
    Ok(parsed)
}

fn verify_provenance(component: &AssetComponent) -> Result<(), LifeAssetError> {
    let fields = [
        ("author", component.provenance.author.trim()),
        ("source_url", component.provenance.source_url.trim()),
        ("source_commit", component.provenance.source_commit.trim()),
        ("license", component.provenance.license.trim()),
        ("license_url", component.provenance.license_url.trim()),
    ];
    for (field, value) in fields {
        if value.is_empty() {
            return Err(LifeAssetError::InvalidProvenance {
                id: component.id.clone(),
                field,
            });
        }
    }
    if !component.provenance.source_url.starts_with("https://") {
        return Err(LifeAssetError::InvalidProvenance {
            id: component.id.clone(),
            field: "source_url",
        });
    }
    let commit = component.provenance.source_commit.as_bytes();
    if !(commit.len() == 40 || commit.len() == 64) || !commit.iter().all(u8::is_ascii_hexdigit) {
        return Err(LifeAssetError::InvalidProvenance {
            id: component.id.clone(),
            field: "source_commit",
        });
    }
    if !component.provenance.license_url.starts_with("https://") {
        return Err(LifeAssetError::InvalidProvenance {
            id: component.id.clone(),
            field: "license_url",
        });
    }
    if !component.provenance.redistribution_permitted {
        return Err(LifeAssetError::RedistributionForbidden(
            component.id.clone(),
        ));
    }
    Ok(())
}

fn verify_metadata(component: &AssetComponent) -> Result<(), LifeAssetError> {
    if component.id.is_empty()
        || !component
            .id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
    {
        return Err(LifeAssetError::InvalidVerification {
            id: component.id.clone(),
            field: "id",
        });
    }
    if component.bounds.min_x > component.bounds.max_x
        || component.bounds.min_y > component.bounds.max_y
    {
        return Err(LifeAssetError::InvalidBounds(component.id.clone()));
    }
    if component.period == 0 || component.phase >= component.period {
        return Err(LifeAssetError::InvalidTiming(component.id.clone()));
    }
    if [
        component.isolation.left,
        component.isolation.right,
        component.isolation.top,
        component.isolation.bottom,
    ]
    .contains(&0)
    {
        return Err(LifeAssetError::InvalidIsolation(component.id.clone()));
    }
    let mut names = BTreeSet::new();
    if component.ports.is_empty() {
        return Err(LifeAssetError::InvalidPort {
            id: component.id.clone(),
            message: "at least one port is required".into(),
        });
    }
    for port in &component.ports {
        if port.name.trim().is_empty() || !names.insert(port.name.as_str()) {
            return Err(LifeAssetError::InvalidPort {
                id: component.id.clone(),
                message: format!("empty or duplicate port name {:?}", port.name),
            });
        }
        if !component.bounds.contains((port.x, port.y)) {
            return Err(LifeAssetError::InvalidPort {
                id: component.id.clone(),
                message: format!("port {:?} is outside bounds", port.name),
            });
        }
        if port.phase >= component.period {
            return Err(LifeAssetError::InvalidPort {
                id: component.id.clone(),
                message: format!("port {:?} phase is outside the period", port.name),
            });
        }
    }
    if !component.verification.independently_verified {
        return Err(LifeAssetError::NotIndependentlyVerified(
            component.id.clone(),
        ));
    }
    for (field, value) in [
        ("verifier", component.verification.verifier.trim()),
        ("method", component.verification.method.trim()),
        ("evidence", component.verification.evidence.trim()),
    ] {
        if value.is_empty() {
            return Err(LifeAssetError::InvalidVerification {
                id: component.id.clone(),
                field,
            });
        }
    }
    Ok(())
}

impl AssetBounds {
    fn contains(self, (x, y): (i64, i64)) -> bool {
        x >= self.min_x && x <= self.max_x && y >= self.min_y && y <= self.max_y
    }
}

fn safe_relative_path(path: &str) -> Result<&Path, LifeAssetError> {
    let path_ref = Path::new(path);
    if path.is_empty()
        || path_ref.is_absolute()
        || path_ref
            .components()
            .any(|part| !matches!(part, Component::Normal(_)))
    {
        return Err(LifeAssetError::UnsafePath(path.to_owned()));
    }
    Ok(path_ref)
}

fn parse_pattern(text: &str, format: PatternFormat) -> Result<Vec<(i64, i64)>, String> {
    match format {
        PatternFormat::Rle => parse_rle(text),
        PatternFormat::Coordinates => parse_coordinates(text),
    }
}

fn parse_coordinates(text: &str) -> Result<Vec<(i64, i64)>, String> {
    let mut cells = BTreeSet::new();
    for (index, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let values = line.split_whitespace().collect::<Vec<_>>();
        if values.len() != 2 {
            return Err(format!(
                "coordinate line {} must contain exactly x and y",
                index + 1
            ));
        }
        let x = values[0]
            .parse::<i64>()
            .map_err(|_| format!("invalid x coordinate on line {}", index + 1))?;
        let y = values[1]
            .parse::<i64>()
            .map_err(|_| format!("invalid y coordinate on line {}", index + 1))?;
        if !cells.insert((x, y)) {
            return Err(format!("duplicate coordinate ({x}, {y})"));
        }
    }
    Ok(cells.into_iter().collect())
}

fn parse_rle(text: &str) -> Result<Vec<(i64, i64)>, String> {
    let mut lines = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'));
    let header = lines
        .next()
        .ok_or_else(|| "missing RLE header".to_owned())?;
    let mut width = None;
    let mut height = None;
    for field in header.split(',') {
        let (key, value) = field
            .split_once('=')
            .ok_or_else(|| "malformed RLE header".to_owned())?;
        match key.trim().to_ascii_lowercase().as_str() {
            "x" => {
                width = Some(
                    value
                        .trim()
                        .parse::<u64>()
                        .map_err(|_| "invalid RLE width".to_owned())?,
                )
            }
            "y" => {
                height = Some(
                    value
                        .trim()
                        .parse::<u64>()
                        .map_err(|_| "invalid RLE height".to_owned())?,
                )
            }
            "rule" if value.trim().eq_ignore_ascii_case("B3/S23") => {}
            "rule" => return Err("only Conway B3/S23 patterns are accepted".into()),
            _ => return Err(format!("unknown RLE header field {:?}", key.trim())),
        }
    }
    let (width, height) = (
        width.ok_or_else(|| "missing RLE width".to_owned())?,
        height.ok_or_else(|| "missing RLE height".to_owned())?,
    );
    if width == 0
        || height == 0
        || width > i64::MAX.unsigned_abs()
        || height > i64::MAX.unsigned_abs()
    {
        return Err("RLE dimensions are out of range".into());
    }
    let body = lines.collect::<String>();
    let mut cells = Vec::new();
    let (mut x, mut y, mut run) = (0_u64, 0_u64, 0_u64);
    let mut terminated = false;
    for byte in body.bytes().filter(|byte| !byte.is_ascii_whitespace()) {
        if terminated {
            return Err("RLE pattern has data after ! terminator".into());
        }
        if byte.is_ascii_digit() {
            run = run
                .checked_mul(10)
                .and_then(|n| n.checked_add(u64::from(byte - b'0')))
                .ok_or_else(|| "RLE run length overflow".to_owned())?;
            continue;
        }
        let count = if run == 0 { 1 } else { run };
        run = 0;
        match byte.to_ascii_lowercase() {
            b'b' | b'o' => {
                let end = x
                    .checked_add(count)
                    .ok_or_else(|| "RLE row overflow".to_owned())?;
                if y >= height || end > width {
                    return Err("RLE data exceeds declared dimensions".into());
                }
                if byte.eq_ignore_ascii_case(&b'o') {
                    let expanded = u64::try_from(cells.len())
                        .map_err(|_| "RLE live-cell count exceeds u64".to_owned())?
                        .checked_add(count)
                        .ok_or_else(|| "RLE live-cell count overflow".to_owned())?;
                    if expanded > MAX_PATTERN_CELLS {
                        return Err(format!("RLE live-cell count exceeds {MAX_PATTERN_CELLS}"));
                    }
                    let live_y =
                        i64::try_from(y).map_err(|_| "RLE y coordinate exceeds i64".to_owned())?;
                    for live_x in x..end {
                        let live_x = i64::try_from(live_x)
                            .map_err(|_| "RLE x coordinate exceeds i64".to_owned())?;
                        cells.push((live_x, live_y));
                    }
                }
                x = end;
            }
            b'$' => {
                y = y
                    .checked_add(count)
                    .ok_or_else(|| "RLE row overflow".to_owned())?;
                x = 0;
                if y > height {
                    return Err("RLE data exceeds declared dimensions".into());
                }
            }
            b'!' if count == 1 => {
                terminated = true;
            }
            b'!' => return Err("RLE terminator cannot have a run length".into()),
            other => return Err(format!("invalid RLE token {:?}", char::from(other))),
        }
    }
    if !terminated {
        return Err("RLE pattern is missing ! terminator".into());
    }
    cells.sort_unstable();
    cells.dedup();
    Ok(cells)
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RequiredExt;

    #[test]
    fn parses_rle_and_coordinate_patterns() {
        assert_eq!(
            parse_rle("#N glider\nx = 3, y = 3, rule = B3/S23\nbob$2bo$3o!"),
            Ok(vec![(0, 2), (1, 0), (1, 2), (2, 1), (2, 2)])
        );
        assert_eq!(
            parse_coordinates("#Life 1.06\n-1 2\n3 4\n"),
            Ok(vec![(-1, 2), (3, 4)])
        );
    }

    #[test]
    fn rejects_non_conway_and_duplicate_coordinates() {
        assert!(parse_rle("x = 1, y = 1, rule = B36/S23\no!").is_err());
        assert!(parse_coordinates("1 2\n1 2\n").is_err());
    }

    #[test]
    fn sha256_matches_standard_vectors() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn rejects_paths_that_can_escape_the_asset_directory() {
        assert!(matches!(
            safe_relative_path("../outside.rle"),
            Err(LifeAssetError::UnsafePath(_))
        ));
        assert!(matches!(
            safe_relative_path("/tmp/outside.rle"),
            Err(LifeAssetError::UnsafePath(_))
        ));
        assert!(safe_relative_path("patterns/clock.rle").is_ok());
    }

    #[test]
    fn checked_in_manifest_remains_fail_closed() {
        let registry =
            AssetRegistry::load_repository().or_invariant("repository manifest should parse");
        assert!(registry.manifest().components.is_empty());
        assert!(matches!(
            registry.verify(),
            Err(LifeAssetError::ManifestBlocked { .. })
        ));
    }

    #[test]
    fn verified_but_incomplete_manifest_fails_closed() {
        let registry = AssetRegistry {
            root: PathBuf::new(),
            manifest: AssetManifest {
                schema: MANIFEST_SCHEMA.into(),
                status: ManifestStatus::Verified,
                components: Vec::new(),
                blocker: None,
            },
        };
        assert!(matches!(
            registry.verify(),
            Err(LifeAssetError::MissingRequiredComponents(kinds))
                if kinds == REQUIRED_V1_COMPONENT_KINDS
        ));
    }

    #[test]
    fn metadata_requires_independent_verification_and_valid_phase() {
        let mut component = sample_component();
        component.verification.independently_verified = false;
        assert!(matches!(
            verify_metadata(&component),
            Err(LifeAssetError::NotIndependentlyVerified(_))
        ));
        component.verification.independently_verified = true;
        component.phase = component.period;
        assert!(matches!(
            verify_metadata(&component),
            Err(LifeAssetError::InvalidTiming(_))
        ));
    }

    #[test]
    fn verifies_pattern_digest_and_cells_from_disk() {
        let root = std::env::temp_dir().join(format!(
            "asimplelife-life-assets-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        fs::create_dir(&root).or_invariant("temporary asset directory should be created");
        let pattern = b"#Life 1.06\n0 0\n2 1\n";
        fs::write(root.join("clock.cells"), pattern)
            .or_invariant("temporary pattern should be written");

        let mut component = sample_component();
        component.pattern.path = "clock.cells".into();
        component.pattern.format = PatternFormat::Coordinates;
        component.pattern.sha256 = sha256_hex(pattern);
        assert_eq!(
            verify_component(&root, &component),
            Ok(vec![(0, 0), (2, 1)])
        );

        component.pattern.sha256 = "0".repeat(64);
        assert!(matches!(
            verify_component(&root, &component),
            Err(LifeAssetError::DigestMismatch { .. })
        ));
        fs::remove_dir_all(root).or_invariant("temporary asset directory should be removed");
    }

    fn sample_component() -> AssetComponent {
        AssetComponent {
            id: "clock".into(),
            kind: ComponentKind::Clock,
            pattern: AssetPattern {
                path: "clock.rle".into(),
                format: PatternFormat::Rle,
                sha256: "0".repeat(64),
            },
            provenance: Provenance {
                author: "Author".into(),
                source_url: "https://example.test/source".into(),
                source_commit: "a".repeat(40),
                license: "MIT".into(),
                license_url: "https://example.test/license".into(),
                redistribution_permitted: true,
            },
            bounds: AssetBounds {
                min_x: 0,
                min_y: 0,
                max_x: 2,
                max_y: 2,
            },
            ports: vec![AssetPort {
                name: "tick".into(),
                x: 1,
                y: 1,
                role: PortRole::Output,
                direction: PortDirection::SouthEast,
                phase: 0,
            }],
            period: 1,
            phase: 0,
            isolation: Isolation {
                left: 1,
                right: 1,
                top: 1,
                bottom: 1,
            },
            verification: VerificationRecord {
                independently_verified: true,
                verifier: "Independent verifier".into(),
                method: "scalar Life evolution".into(),
                evidence: "https://example.test/evidence".into(),
            },
        }
    }
}
