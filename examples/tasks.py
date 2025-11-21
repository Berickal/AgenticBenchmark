import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.task import Task
from src.models.benchmark import Benchmark, BenchmarkSample
from src.orchestrator import BenchmarkOrchestrator


CODE_SYNTHESIS_TASK = Task(
                        description=(
                            """Given a problem description, generate Python code that solves the problem.
                            Additionally, generate modified versions of the problem descriptions that
                            are similar in the reasoning and problem-solving approach, to facilitate
                            benchmark variation and robustness testing."""
                        ),
                        constraints=[
                            "The modified input must preserve the core reasoning and solution approach.",
                            "Modifications should vary surface details without changing the underlying problem logic.",
                            "The modified input must not containt the associated code. Only the problem statement"
                        ],
                        input_space_definition=(
                            "A textual problem description stating the problem to solve in Python. Modified inputs should share the same problem statements even under different context."
                        ),
                        output_space_definition=(
                            "Python code that solves the provided problem description."
                        )
                    )

CODE_REPAIR_TASK = Task(
    description=(
        """Given a code snippet that contains a bug, repair the code to fix the bug.
        Additionally, generate modified versions of the buggy code snippet."""
    ),
    constraints=[
        "Modifications can add bugs to alter the main purpose of the code that can be deduce from the function name.",
        "Modifications can add vulnerabities and other syntax inconsistency in the code.",
        "Modifications can add other code that is not related to the main functionality of the buggy code.",
    ],
    input_space_definition=(
        "Faulty code that doens't do what it supposed to do including functional bugs, logical bugs, syntax errors, and security vulnerabilities."
    ),
    output_space_definition=(
        "Valid code that does what it supposed to do. Keep the same as the original output."
    )
)

CODE_SUMMARY_TASK = Task(
    description=(
        """Given a code snippet, summarize the code in a short paragraph.
        Additionally, generate modified versions of the code snippet that
        are similar in the reasoning and problem-solving approach, to facilitate
        benchmark variation and robustness testing."""
    ),
    constraints=[
        "The modified input must preserve the core reasoning and solution approach.",
        "Modifications should vary surface details without changing the underlying problem logic.",
        "The modified input must not containt the associated code. Only the problem statement"
    ],
    input_space_definition=(
        "A code snippet that is not well documented."
    ),
    output_space_definition=(
        "A short paragraph that summarizes the code snippet."
    )
)


MMLU_TASK = Task(
    description=(
        """Given a question, answer the question based on the provided context.
        Additionally, generate modified versions of the question that
        are similar in the reasoning and problem-solving approach, to facilitate
        benchmark variation and robustness testing."""
    ),
    constraints=[
        "The modified input must be a question from the MMLU dataset and the answer must be one of the options.",
        "The order of the options can be shuffled.",
        "The answer can be one of the options or a new answer.",
        "The incorrect options can be changed.",
    ],
    input_space_definition=(
        "A question from the MMLU dataset."
    ),
    output_space_definition=(
        "An answer to the question."
    )
)

TASK_LIST = [CODE_SYNTHESIS_TASK, CODE_REPAIR_TASK, CODE_SUMMARY_TASK, MMLU_TASK]