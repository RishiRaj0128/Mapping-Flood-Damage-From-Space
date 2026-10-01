"""Exceptions for compliance and dataset validation violations."""


class ComplianceViolationError(Exception):
    """Base exception for all regulatory and competition rule violations."""
    pass


class BannedDatasetError(ComplianceViolationError):
    """Raised when an unauthorized dataset (e.g. WorldPop, GHSL, CEMS) is requested as input."""
    pass


class OSMDateViolationError(ComplianceViolationError):
    """Raised when an OpenStreetMap query uses a timestamp after the 2026-07-27 cutoff."""
    pass


class CrossTrackPairError(ComplianceViolationError):
    """Raised when Sentinel-1 pair does not match the same relative orbit and flight direction."""
    pass


class ForbiddenTerminologyError(ComplianceViolationError):
    """Raised when unpermitted absolute damage terminology (e.g. 'destroyed') is used."""
    pass
