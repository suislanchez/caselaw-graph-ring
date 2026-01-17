"""Pydantic models for legal case data."""

from datetime import datetime
from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class Court(BaseModel):
    """Court information."""
    id: Optional[int] = None
    slug: str
    name: str
    name_abbreviation: Optional[str] = None


class Opinion(BaseModel):
    """Individual opinion within a case."""
    type: str  # "majority", "dissent", "concurrence", etc.
    author: Optional[str] = None
    text: str


class CaseBody(BaseModel):
    """Full case body content."""
    opinions: List[Opinion] = Field(default_factory=list)

    @property
    def full_text(self) -> str:
        """Concatenate all opinions into full text."""
        return "\n\n".join(op.text for op in self.opinions)

    @property
    def majority_opinion(self) -> Optional[str]:
        """Get majority opinion text."""
        for op in self.opinions:
            if op.type == "majority":
                return op.text
        # Fall back to first opinion if no majority labeled
        return self.opinions[0].text if self.opinions else None


class Case(BaseModel):
    """Core case model for the pipeline."""
    id: str
    name: str
    name_abbreviation: Optional[str] = None
    date: datetime
    court: Court
    text: str  # Full opinion text
    outcome: Optional[Literal["petitioner", "respondent"]] = None
    citations_raw: List[str] = Field(default_factory=list)

    # Additional metadata
    docket_number: Optional[str] = None
    citations: List[str] = Field(default_factory=list)  # Official citations like "123 U.S. 456"
    cap_id: Optional[int] = None  # Original CAP API ID
    scdb_id: Optional[str] = None  # SCDB case ID if matched

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class CAPCase(BaseModel):
    """Raw case from Caselaw Access Project API."""
    id: int
    url: Optional[str] = None
    name: str
    name_abbreviation: Optional[str] = None
    decision_date: str
    docket_number: Optional[str] = None
    first_page: Optional[str] = None
    last_page: Optional[str] = None
    citations: List[dict] = Field(default_factory=list)
    court: dict
    jurisdiction: Optional[dict] = None
    casebody: Optional[dict] = None

    def to_case(self) -> Case:
        """Convert CAP case to our Case model."""
        # Parse date
        try:
            date = datetime.strptime(self.decision_date, "%Y-%m-%d")
        except ValueError:
            date = datetime.now()

        # Extract court
        court = Court(
            id=self.court.get("id"),
            slug=self.court.get("slug", "unknown"),
            name=self.court.get("name", "Unknown Court"),
            name_abbreviation=self.court.get("name_abbreviation")
        )

        # Extract text from casebody
        text = ""
        if self.casebody and "data" in self.casebody:
            data = self.casebody["data"]
            if "opinions" in data:
                opinions = data["opinions"]
                text = "\n\n".join(op.get("text", "") for op in opinions)
            elif "text" in data:
                text = data["text"]

        # Extract citation strings
        citation_strs = [c.get("cite", "") for c in self.citations if c.get("cite")]

        return Case(
            id=str(self.id),
            cap_id=self.id,
            name=self.name,
            name_abbreviation=self.name_abbreviation,
            date=date,
            court=court,
            text=text,
            docket_number=self.docket_number,
            citations=citation_strs,
            citations_raw=[]  # Will be populated by citation extractor
        )


class SCDBCase(BaseModel):
    """Supreme Court Database case record."""
    case_id: str = Field(alias="caseId")
    docket: Optional[str] = None
    case_name: str = Field(alias="caseName")
    date_decision: Optional[str] = Field(None, alias="dateDecision")
    us_cite: Optional[str] = Field(None, alias="usCite")
    s_ct_cite: Optional[str] = Field(None, alias="sctCite")
    l_ed_cite: Optional[str] = Field(None, alias="ledCite")
    lexis_cite: Optional[str] = Field(None, alias="lexisCite")
    party_winning: Optional[int] = Field(None, alias="partyWinning")
    decision_direction: Optional[int] = Field(None, alias="decisionDirection")
    issue_area: Optional[int] = Field(None, alias="issueArea")

    class Config:
        populate_by_name = True

    @property
    def outcome(self) -> Optional[Literal["petitioner", "respondent"]]:
        """Convert partyWinning to outcome label."""
        if self.party_winning == 1:
            return "petitioner"
        elif self.party_winning == 0:
            return "respondent"
        return None

    @property
    def primary_citation(self) -> Optional[str]:
        """Get the primary citation for matching."""
        return self.us_cite or self.s_ct_cite or self.l_ed_cite


class DataSplit(BaseModel):
    """Train/val/test split container."""
    train: List[Case]
    val: List[Case]
    test: List[Case]

    @property
    def total_count(self) -> int:
        return len(self.train) + len(self.val) + len(self.test)


class CaseMetadata(BaseModel):
    """Metadata for a legal case, used for indexing and matching."""
    id: str
    name: str
    date: datetime
    court: str
    citation: str  # Official citation (e.g., "347 U.S. 483")
    scdb_id: Optional[str] = None  # SCDB case ID if available

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class DatasetStats(BaseModel):
    """Statistics about the dataset."""
    total_cases: int
    cases_with_outcome: int
    petitioner_wins: int
    respondent_wins: int
    avg_text_length: float
    avg_citations_per_case: float
    courts: dict  # court_name -> count
    date_range: tuple  # (min_date, max_date)
