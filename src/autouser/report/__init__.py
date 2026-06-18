"""Report generation: transforms step results into actionable friction logs."""

from autouser.report.models import FrictionLog, TaskOutcome
from autouser.report.generator import ReportGenerator

__all__ = ["FrictionLog", "TaskOutcome", "ReportGenerator"]
