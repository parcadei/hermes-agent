-- Hermes NLCDM: NonLinear Coupled Dynamical Memory
-- V2 proof system for coupled memory dynamics
--
-- Phase 1: Energy Well-Formedness
-- Phase 2: Single Update Convergence
-- Phase 3: Pairwise Coupling
-- Phase 4: Conditional Capacity
-- Phase 5: Modular Structure
-- Phase 5.5: Block-Diagonal Decomposition
-- Phase 5.7: Weight Perturbation
-- Phase 5.8: Temperature-Parametric Dynamics
-- Phase 5.9: Spurious State Characterization
-- Phase 6: Full System Lyapunov

-- Phase 1
import HermesNLCDM.Energy
import HermesNLCDM.EnergyBounds
import HermesNLCDM.LocalMinima

-- Phase 2
import HermesNLCDM.Dynamics

-- Phase 3
import HermesNLCDM.Coupling

-- Phase 4
import HermesNLCDM.Capacity

-- Phase 5
import HermesNLCDM.Modular

-- Phase 5.5
import HermesNLCDM.BlockDiagonal

-- Phase 5.7
import HermesNLCDM.WeightUpdate

-- Phase 5.8
import HermesNLCDM.Temperature

-- Phase 5.9
import HermesNLCDM.SpuriousStates

-- Phase 5.9b
import HermesNLCDM.EnergyGap

-- Phase 5D.1
import HermesNLCDM.BasinVolume

-- Phase 6
import HermesNLCDM.Lyapunov

-- Phase 7: Reasoning Trace Centroid Energy Wells
import HermesNLCDM.TraceCentroid

-- Phase 8: Bridge Formation via Trace Centroids
import HermesNLCDM.BridgeFormation

-- Phase 9a: V1↔V2 Bridge Monotonicity Chain
import HermesNLCDM.MonotonicityChain

-- Phase 9b: Conditional Bridge Monotonicity Under Dream Operations
import HermesNLCDM.ConditionalMonotonicity

-- Phase 10: Cross-Domain Transfer via Bridge Patterns
import HermesNLCDM.TransferDynamics

-- Phase 11: REM-Explore Perturbation-Response Mechanism
import HermesNLCDM.PerturbationResponse
