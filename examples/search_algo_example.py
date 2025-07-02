# Standard library imports
import os
import time
import argparse
from typing import Any, Tuple

# Third-party imports
import datasets
import numpy as np

# Opto imports
from opto import trace
from opto.optimizers import OptoPrime
from opto.optimizers.utils import print_color
from opto.trace.modules import Module
from opto.trainer.algorithms.basic_algorithms import MinibatchAlgorithm, BasicSearchAlgorithm
from opto.trainer.algorithms.beamsearch_algorithm import BeamsearchAlgorithm, BeamsearchHistoryAlgorithm
from opto.trainer.algorithms.UCBsearch import UCBSearchAlgorithm
from opto.trainer.guide import AutoGuide
from opto.trainer.loggers import DefaultLogger
from opto.utils.llm import LLM

# Set default model
# os.environ["TRACE_LITELLM_MODEL"] = "vertex_ai/gemini-2.0-flash"

@trace.model
class Learner(Module):
    """A basic LLM Agent for solving math problems."""
    
    def __init__(self, 
                system_prompt: str = "You're a helpful agent answering math problems.",
                user_prompt_template: str = "Solve the following math problem step-by-step: {message}",
                llm: LLM = None):
        """Initialize the learner agent.
        
        Args:
            system_prompt: System prompt to guide LLM behavior
            user_prompt_template: Template for formatting user messages
            llm: LLM instance to use for generation (defaults to gpt-3.5-turbo)
        """
        super().__init__()
        self.system_prompt = trace.node(system_prompt, trainable=True)
        self.user_prompt_template = trace.node(user_prompt_template, trainable=True)
        self.llm = llm or LLM(model="gpt-3.5-turbo")

    @trace.bundle()
    def call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Call LLM model with the given prompts.
        
        Args:
            system_prompt: The system prompt
            user_prompt: The user prompt
            
        Returns:
            The LLM response content
        """
        response = self.llm(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        return response.choices[0].message.content

    def forward(self, message: Any) -> str:
        """Agent's forward pass to process a message.
        
        Args:
            message: The input message to process
            
        Returns:
            The generated response
        """ 
        user_prompt = self.user_prompt_template.format(message=message)
        return self.call_llm(self.system_prompt, user_prompt)


class TeacherGuide(AutoGuide):
    """Guide that uses LLM to judge answers and provide feedback."""
    
    def __init__(self, model: str = "gpt-4o-mini"):
        """Initialize the teacher guide.
        
        Args:
            model: The LLM model to use for evaluation
        """
        super().__init__()
        self.guide_llm = LLM(model=model)
        self.system_prompt = "You are an expert math teacher evaluating student answers."
        self.judge_prompt_template = (
            "Carefully review the following three distinct sections:\n\n"
            "SECTION 1: The Math Problem\n"
            "----------------------------\n"
            "{query}\n"
            "----------------------------\n\n"
            "SECTION 2: The Student's Full Answer\n"
            "----------------------------\n"
            "{response}\n"
            "----------------------------\n\n"
            "SECTION 3: The Official Correct Answer\n"
            "----------------------------\n"
            "{reference}\n"
            "----------------------------\n\n"
            "INSTRUCTIONS FOR JUDGING:\n"
            "1. Your primary task is to compare the student's **final numerical result** (or final conclusion if no number is present) from SECTION 2 with the **Official Correct Answer** provided in SECTION 3.\n"
            "2. When evaluating SECTION 2 (Student's Full Answer), focus SOLELY on the **final answer part** of the student's response. Ignore all intermediate steps, reasoning, or explanations for the correctness check unless the problem specifically asks for reasoning as the final answer.\n"
            "3. Determine if the student's **final answer** is equivalent to the **Official Correct Answer**.\n\n"
            "RESPONSE FORMAT:\n"
            "- If the student's final answer (from SECTION 2) IS equivalent to the Official Correct Answer (from SECTION 3), respond ONLY with the exact phrase: 'Correct [TERMINATE]'\n"
            "- If the student's final answer IS NOT equivalent, respond ONLY with specific and actionable feedback. The feedback should clearly explain the error in the student's final answer and guide them on how to arrive at the Official Correct Answer."
        )

    def get_feedback(self, task: str, response: str, info: Any, **kwargs) -> Tuple[float, str]:
        """Get feedback on a student response.
        
        Args:
            task: The original math problem
            response: The student's answer
            info: The reference/correct answer
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (score, feedback_text)
        """
        user_prompt = self.judge_prompt_template.format(
            query=task,
            response=response,
            reference=info
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        llm_response = self.guide_llm(messages=messages)
        feedback_text = llm_response.choices[0].message.content

        if 'Correct [TERMINATE]' in feedback_text:
            return 1.0, "Correct."
        else:
            return 0.0, f"Incorrect. Feedback: {feedback_text}"
    
    def metric(self, task: str, content: str, info: Any, **kwargs) -> float:
        """Calculate the metric score for an answer.
        
        Args:
            task: The original math problem
            content: The student's answer
            info: The reference/correct answer
            **kwargs: Additional arguments
            
        Returns:
            Score (0.0 or 1.0)
        """
        score, _ = self.get_feedback(task, content, info, **kwargs)
        return score


class SimpleLogger(DefaultLogger):
    """Simplified logger that only shows important metrics."""
    
    def log(self, name: str, data: Any, step: int, **kwargs):
        """Log only specific metrics to reduce output clutter.
        
        Args:
            name: The name of the metric
            data: The metric value
            step: The current step
            **kwargs: Additional logging arguments
        """
        important_metrics = [
            'Average train score',
            'Average test score',
            'Validation score'
        ]
        
        if name in important_metrics or 'Parameter' in name:
            super().log(name, data, step, **kwargs)


def main():
    """Run the main training process with command line arguments."""
    parser = argparse.ArgumentParser(description='Train agent using various algorithms')
    
    # Algorithm parameters
    parser.add_argument('--algorithm_type', type=str, default='UCBsearch',
                       choices=['minibatch', 'basicsearch', 'beamsearch', 'beamsearchhistory', 'UCBsearch'],
                       help='Type of algorithm to use')
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='xuanfeiren/math_hard_gemini',
                       help='Dataset to use for training')
    parser.add_argument('--num_train_samples', type=int, default=66,
                       help='Number of training samples')
    parser.add_argument('--num_validate_samples', type=int, default=20,
                       help='Number of validation samples')
    parser.add_argument('--num_test_samples', type=int, default=20,
                       help='Number of test samples')
    
    # LLM Model parameters
    parser.add_argument('--trace_model', type=str, default=None,
                       help='Model to use for trace operations')
    parser.add_argument('--student_model', type=str, default=None,
                       help='Model to use for student agent')
    parser.add_argument('--teacher_model', type=str, default=None,
                       help='Model to use for teacher guide')
    
    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=1,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2,
                       help='Training batch size')
    parser.add_argument('--num_threads', type=int, default=10,
                       help='Number of threads for parallel processing')
    parser.add_argument('--eval_frequency', type=int, default=2,
                       help='How often to run evaluation')
    parser.add_argument('--log_frequency', type=int, default=20,
                       help='How often to log results')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    # Algorithm-specific parameters
    parser.add_argument('--beam_width', type=int, default=3,
                       help='Beam width for beam search algorithms')
    parser.add_argument('--num_proposals', type=int, default=2,
                       help='Number of proposals for beam search algorithms')
    parser.add_argument('--max_depth', type=int, default=20,
                       help='Maximum depth for beam search algorithms')
    parser.add_argument('--validation_dataset_size', type=int, default=20,
                       help='Size of validation dataset for beam search')
    parser.add_argument('--max_history_size', type=int, default=12,
                       help='Maximum history size for history-based algorithms')
    parser.add_argument('--num_basicsearch_proposals', type=int, default=2,
                       help='Number of proposals for basic search algorithm')
    
    # UCB algorithm-specific parameters
    parser.add_argument('--max_buffer_size', type=int, default=10,
                       help='Maximum buffer size for UCB algorithms')
    parser.add_argument('--ucb_exploration_factor', type=float, default=1.0,
                       help='UCB exploration factor')
    parser.add_argument('--num_search_iterations', type=int, default=100,
                       help='Number of search iterations for UCB algorithms')
    parser.add_argument('--train_batch_size_ucb', type=int, default=2,
                       help='Training batch size for UCB algorithms')
    parser.add_argument('--evaluation_batch_size', type=int, default=20,
                       help='Evaluation batch size for UCB algorithms')
    
    args = parser.parse_args()
    
    # Set environment variables
    if args.trace_model:
        os.environ["TRACE_LITELLM_MODEL"] = args.trace_model

    # Set random seed
    np.random.seed(args.seed)
    
    # Check for API Keys
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        print_color("Warning: OPENAI_API_KEY or ANTHROPIC_API_KEY environment variables not found. LLM calls may fail.", "red")

    # Load and prepare data
    print(f"Loading data from {args.dataset}...")
    math_data = datasets.load_dataset(args.dataset)
    
    # Select data subsets
    train_data = math_data['train'].select(
        range(args.num_train_samples, args.num_train_samples + args.num_validate_samples)
    )
    validate_data = train_data
    test_data = math_data['test'].select(range(args.num_test_samples))

    # Format data for trainer
    train_dataset = {'inputs': train_data['problem'], 'infos': train_data['solution']}
    validate_dataset = {'inputs': validate_data['problem'], 'infos': validate_data['solution']}
    test_dataset = {'inputs': test_data['problem'], 'infos': test_data['solution']}
    
    # Log dataset sizes
    print(f"Training samples: {len(train_dataset['inputs'])}")
    print(f"Validation samples: {len(validate_dataset['inputs'])}")
    print(f"Test samples: {len(test_dataset['inputs'])}")

    # Initialize components
    print("Initializing Agent, Guide, Optimizer, Algorithm...")
    student_llm = LLM(model=args.student_model)
    agent = Learner(llm=student_llm)

    train_guide = TeacherGuide(model=args.teacher_model)
    validate_guide = TeacherGuide(model=args.teacher_model)

    optimizer = OptoPrime(agent.parameters())
    logger = SimpleLogger()
    
    # Create algorithm
    if args.algorithm_type == 'minibatch':
        algorithm = MinibatchAlgorithm(
            agent=agent,
            optimizer=optimizer,
            logger=logger,
            num_threads=args.num_threads
        )
    elif args.algorithm_type == 'basicsearch':
        algorithm = BasicSearchAlgorithm(
            agent=agent,
            optimizer=optimizer,
            logger=logger,
            num_threads=args.num_threads
        )
    elif args.algorithm_type == 'beamsearch':
        algorithm = BeamsearchAlgorithm(
            agent=agent,
            optimizer=optimizer,
            logger=logger,
            num_threads=args.num_threads
        )
    elif args.algorithm_type == 'beamsearchhistory':
        algorithm = BeamsearchHistoryAlgorithm(
            agent=agent,
            optimizer=optimizer,
            logger=logger,
            num_threads=args.num_threads
        )
    elif args.algorithm_type == 'UCBsearch':
        algorithm = UCBSearchAlgorithm(
            agent=agent,
            optimizer=optimizer,
            logger=logger,
            num_threads=args.num_threads,
            max_buffer_size=args.max_buffer_size,
            ucb_exploration_factor=args.ucb_exploration_factor
        )
    else:
        raise ValueError(f"Unknown algorithm type: {args.algorithm_type}")
    
    # Prepare training parameters
    train_params = {
        "guide": train_guide,
        "train_dataset": train_dataset,
        "num_epochs": args.num_epochs,
        "num_threads": args.num_threads,
        "batch_size": args.batch_size,
        "test_dataset": test_dataset,
        "validate_dataset": validate_dataset,
        "validate_guide": validate_guide,
        "eval_frequency": args.eval_frequency,
        "log_frequency": args.log_frequency,
        "validation_dataset_size": args.validation_dataset_size,
    }
    
    # Add algorithm-specific parameters
    if args.algorithm_type in ['beamsearch', 'beamsearchhistory']:
        train_params.update({
            "beam_width": args.beam_width,
            "num_proposals": args.num_basicsearch_proposals,
            "max_depth": args.max_depth
        })
        
        if args.algorithm_type == 'beamsearchhistory':
            train_params["max_history_size"] = args.max_history_size
            
    elif args.algorithm_type == 'basicsearch':
        train_params["num_proposals"] = args.num_basicsearch_proposals
    
    elif args.algorithm_type == 'UCBsearch':
        train_params.update({
            "num_search_iterations": args.num_search_iterations,
            "train_batch_size": args.train_batch_size_ucb,
            "evaluation_batch_size": args.evaluation_batch_size
        })
    
    # Start training
    print(f"Training with {args.algorithm_type} algorithm...")
    start_time = time.time()
    metrics, final_score = algorithm.train(**train_params)
    duration = time.time() - start_time
    print(f"Training complete, time taken: {duration:.2f} seconds")
    
    # Print metrics summary based on algorithm type
    if args.algorithm_type in ['beamsearch', 'beamsearchhistory'] and 'best_validation_scores' in metrics:
        print("\nBest validation scores at each depth:")
        for depth, score in enumerate(metrics['best_validation_scores']):
            print(f"  Depth {depth+1}: {score:.4f}")
    
    elif args.algorithm_type == 'UCBsearch':
        print("\nUCB Algorithm Metrics:")
        if 'best_candidate_scores' in metrics and metrics['best_candidate_scores']:
            print(f"  Best candidate scores over iterations: {len(metrics['best_candidate_scores'])} recorded")
            print(f"  Final best candidate score: {metrics['best_candidate_scores'][-1]:.4f}")
        if 'buffer_avg_score' in metrics and metrics['buffer_avg_score']:
            print(f"  Final buffer average score: {metrics['buffer_avg_score'][-1]:.4f}")
    
    print(f"Final score: {final_score:.4f}")
    
    return metrics, final_score


if __name__ == "__main__":
    main()