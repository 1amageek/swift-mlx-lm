import GGUFToolingCore
import MLXLM

public protocol GGUFFamilyValidator: Sendable {
    var family: DetectedArchitecture { get }
    func validate(context: ValidationContext) -> [GGUFValidationIssue]
    func repairActions(context: ValidationContext, mode: RepairPlanningMode) -> [GGUFRepairAction]
}
