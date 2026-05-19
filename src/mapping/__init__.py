"""Semantic mapping module: depth → top-down occupancy grid."""

from mapping.builder import MapConfig, SemanticMapBuilder
from mapping.visualize import visualize_map

__all__ = ["MapConfig", "SemanticMapBuilder", "visualize_map"]
