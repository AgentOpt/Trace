import numpy as np
import copy
import time
import math
import json # For LLM output parsing
import re # For smart quote replacement
from collections import deque
from typing import Union, List, Tuple, Dict, Any, Optional
import random # Added for alpha probability

from opto import trace
from opto.trainer.utils import async_run # Assuming print_color is in utils
from opto.optimizers.utils import print_color
from opto.trainer.algorithms.basic_algorithm import MinibatchAlgorithm, evaluate, batchify # evaluate and batchify might be useful
from opto.utils.llm import LiteLLM # For the selector LLM

from opto.trace.nodes import ParameterNode
import warnings
from black import format_str, FileMode


def smart_quote_replacement(text: str) -> str:
    """
    Intelligently replace single quotes with double quotes for JSON parsing.
    Handles the specific case where we have mixed quotes like:
    {'key': "value with 'nested' quotes"}
    """
    # For the specific pattern we're seeing, let's handle it step by step:
    
    # Step 1: Replace single quotes around keys
    # Pattern: 'key': -> "key":
    text = re.sub(r"'([^']*?)'(\s*:)", r'"\1"\2', text)
    
    # Step 2: For values that start with double quotes and contain single quotes,
    # we need to escape the internal single quotes or convert them properly
    
    # Let's try a more direct approach for the problematic case:
    # Find patterns like: "text with 'word' more text"
    # We need to escape the internal single quotes
    def escape_internal_quotes(match):
        content = match.group(1)
        # Replace single quotes inside with escaped single quotes
        # Actually, for JSON we can leave single quotes as-is inside double quotes
        return f'"{content}"'
    
    # Replace the pattern: : "content with 'quotes'" -> : "content with 'quotes'"
    # (This should already be valid JSON)
    
    # The main issue is with the outer structure, let's fix that:
    # If the string starts/ends with single quotes around the whole thing
    text = text.strip()
    if text.startswith("{'") and text.endswith("'}"):
        # Replace the outer single quotes but preserve the content
        # This is the pattern: {'str0': "content", 'str1': "more content"}
        text = '{"' + text[2:-2] + '"}'
    
    return text


class UCBSearchAlgorithm(MinibatchAlgorithm):
    """
    UCB Search Algorithm.

    Keeps a buffer of candidates with their statistics (score sum, evaluation count).
    In each iteration:
    1. Picks a candidate 'a' from the buffer with the highest UCB score.
    2. Updates the optimizer with 'a's parameters.
    3. Draws a minibatch from the training set, performs a forward/backward pass, and calls optimizer.step() to get a new candidate 'a''.
    4. Evaluates 'a'' on a validation set minibatch.
    5. Updates statistics of 'a' (based on the training minibatch).
    6. Adds 'a'' (with its validation stats) to the buffer.
    7. If the buffer is full, evicts the candidate with the lowest UCB score.
    """

    def __init__(self,
                 agent: trace.Module,
                 optimizer,
                 max_buffer_size: int = 10,
                 ucb_exploration_factor: float = 1.0,
                 logger=None,
                 num_threads: int = None,
                 *args,
                 **kwargs):
        super().__init__(agent, optimizer, num_threads=num_threads, logger=logger, *args, **kwargs)
        
        self.buffer = deque(maxlen=max_buffer_size) 
        self.max_buffer_size = max_buffer_size
        self.ucb_exploration_factor = ucb_exploration_factor
        
        # To ensure optimizer_step can be called with bypassing=True if needed.
        # This depends on the specific optimizer's implementation.
        # For now, we assume the optimizer has a step method that can return parameters.
        if not hasattr(self.optimizer, 'step'):
            raise ValueError("Optimizer must have a 'step' method.")

        self._total_evaluations_tracker = 0 # Tracks total number of individual candidate evaluations used in UCB calculation for log(T)
        self._candidate_id_counter = 0

    def _sample_minibatch(self, dataset: Dict[str, List[Any]], batch_size: int) -> Tuple[List[Any], List[Any]]:
        """Sample a minibatch from the dataset."""
        if not dataset or not dataset.get('inputs') or not dataset.get('infos'):
            print_color("Warning: Attempted to sample from an empty or malformed dataset.", color='yellow')
            return [], []
        
        dataset_size = len(dataset['inputs'])
        if dataset_size == 0:
            print_color("Warning: Dataset is empty, cannot sample minibatch.", color='yellow')
            return [], []

        actual_batch_size = min(batch_size, dataset_size)
        indices = np.random.choice(dataset_size, actual_batch_size, replace=False)
        xs = [dataset['inputs'][i] for i in indices]
        infos = [dataset['infos'][i] for i in indices]
        return xs, infos

    def _evaluate_candidate(self, 
                              params_to_eval_dict: Dict[str, Any], 
                              dataset: Dict[str, List[Any]], # Changed from validate_dataset
                              guide, # Changed from validate_guide
                              evaluation_batch_size: int, # New parameter name
                              num_threads: Optional[int] = None
                              ) -> Tuple[float, int]:
        """Evaluates a given set of parameters on samples from the provided dataset (now typically train_dataset)."""
        if not dataset or not dataset.get('inputs') or not dataset.get('infos') or not dataset['inputs']:
            print_color("Evaluation dataset is empty or invalid. Returning score -inf, count 0.", color='yellow')
            return -np.inf, 0

        original_params = {p: copy.deepcopy(p.data) for p in self.agent.parameters()}
        self.optimizer.update(params_to_eval_dict)      

        eval_xs, eval_infos = self._sample_minibatch(dataset, evaluation_batch_size) # Use evaluation_batch_size
        
        if not eval_xs:
            print_color("Evaluation minibatch is empty. Returning score -inf, count 0.", color='yellow')
            self.optimizer.update(original_params) 
            return -np.inf, 0

        eval_scores = evaluate(self.agent,
                               guide, # Use main guide
                               eval_xs,
                               eval_infos,
                               min_score=self.min_score if hasattr(self, 'min_score') else None,
                               num_threads=num_threads or self.num_threads,
                               description=f"Evaluating candidate")

        self.optimizer.update(original_params) 

        avg_score = np.mean(eval_scores) if eval_scores and all(s is not None for s in eval_scores) else -np.inf
        eval_count = len(eval_xs) 
        
        return float(avg_score), eval_count

    def _calculate_ucb(self, candidate_buffer_entry: Dict, total_tracked_evaluations: int) -> float:
        """Calculates UCB score for a candidate in the buffer."""
        if candidate_buffer_entry['eval_count'] == 0:
            return float('inf')  # Explore unvisited states first
        
        mean_score = candidate_buffer_entry['score_sum'] / candidate_buffer_entry['eval_count']
        
        # Add 1 to total_tracked_evaluations to prevent log(0) if it's the first evaluation overall
        # and to ensure log argument is > 0.
        # Add 1 to eval_count in denominator as well to ensure it's robust if eval_count is small.
        if total_tracked_evaluations == 0: # Should not happen if we init with one eval
             total_tracked_evaluations = 1
        
        exploration_term = self.ucb_exploration_factor * \
                           math.sqrt(math.log(total_tracked_evaluations) / candidate_buffer_entry['eval_count'])
        
        return mean_score + exploration_term

    def _update_buffer_ucb_scores(self):
        """Recalculates and updates UCB scores for all candidates in the buffer."""
        if not self.buffer:
            return
        
        for candidate_entry in self.buffer:
            candidate_entry['ucb_score'] = self._calculate_ucb(candidate_entry, self._total_evaluations_tracker)

    def train(self,
              guide,  # Guide for train_dataset (feedback generation AND evaluation)
              train_dataset: Dict[str, List[Any]],
              *,
              num_search_iterations: int = 100,
              train_batch_size: int = 2, 
              evaluation_batch_size: int = 20, # Renamed from validation_batch_size, used for all explicit evaluations
              eval_frequency: int = 1, 
              log_frequency: Optional[int] = None,
              save_frequency: Optional[int] = None,
              save_path: str = "checkpoints/ucb_agent.pkl",
              min_score_for_agent_update: Optional[float] = None, # Renamed from min_score to avoid conflict with evaluate's min_score
              verbose: Union[bool, str] = False,
              num_threads: Optional[int] = None,
              **kwargs
              ) -> Tuple[Dict[str, Any], float]: # Returns metrics and best score
        """
        Main training loop for UCB Search Algorithm.
        """
        num_threads = num_threads or self.num_threads
        log_frequency = log_frequency or eval_frequency
        self.min_score = min_score_for_agent_update # Used by parent's evaluate if called, or our own _evaluate_candidate
        total_samples = 0

        # Metrics tracking
        metrics = {
            'best_candidate_scores': [], # Score of the best candidate (e.g., highest mean) found so far at each iteration
            'selected_action_ucb': [], # UCB score of the selected action 'a'
            'new_candidate_scores': [], # Score of the new candidate 'a_prime'
            'buffer_avg_score': [],
            'buffer_avg_evals': [],
        }

# 0. Evaluate the initial parameter on samples of the validation set and add it to the buffer.
        print_color("Evaluating initial parameters using train_dataset samples...", 'cyan')
        initial_params_dict = {p: copy.deepcopy(p.data) for p in self.agent.parameters()}
        initial_score, initial_evals = self._evaluate_candidate(
            initial_params_dict, train_dataset, guide, evaluation_batch_size, num_threads # Use train_dataset and guide
        )
        self._total_evaluations_tracker += initial_evals 
        total_samples += initial_evals

        initial_candidate_entry = {
            'params': initial_params_dict,
            'score_sum': initial_score * initial_evals if initial_score > -np.inf else 0, # Store sum for accurate mean later
            'eval_count': initial_evals,
            'ucb_score': 0.0, # Will be updated
            'iteration_created': 0
        }
        self.buffer.append(initial_candidate_entry)
        self._update_buffer_ucb_scores() # Update UCB for the initial candidate
        print_color(f"Initial candidate: Score {initial_score:.4f}, Evals {initial_evals}", 'yellow')

        # Main search loop
        for iteration in range(1, num_search_iterations + 1):
            if not self.buffer:
                print_color("Buffer is empty, stopping search.", 'red')
                break

            # 1. Pick the candidate 'a' with the highest UCB from the buffer
            self._update_buffer_ucb_scores() # Ensure UCB scores are fresh
            action_candidate_a = self.select(self.buffer)
            
            
            print_color(f"Iter {iteration}/{num_search_iterations}: ", 'blue')
            

            # 2. Load parameters of 'a' into the agent for the optimizer update step
            self.optimizer.update(action_candidate_a['params'])

            # 3. Draw minibatch from the training set, do update from 'a' to get 'a_prime'
            train_xs, train_infos = self._sample_minibatch(train_dataset, train_batch_size)
            if not train_xs:
                print_color(f"Iter {iteration}: Training minibatch empty, skipping optimizer step.", 'yellow')
                continue 

            # Perform forward pass and get feedback for agent parameters 'a'
            outputs_for_a = []
            use_asyncio = self._use_asyncio(num_threads)
            if use_asyncio:
                outputs_for_a = async_run([self.forward]*len(train_xs),
                                   [(self.agent, x, guide, info) for x, info in zip(train_xs, train_infos)],
                                   max_workers=num_threads,
                                   description=f"Iter {iteration}: Forward pass for action 'a' ")
            else:
                outputs_for_a = [self.forward(self.agent, x, guide, info) for x, info in zip(train_xs, train_infos)]

            scores_from_train, targets_from_train, feedbacks_from_train = [], [], []
            for target, score, feedback in outputs_for_a:
                scores_from_train.append(score)
                targets_from_train.append(target)
                feedbacks_from_train.append(feedback)
            
            if not scores_from_train: # Should not happen if train_xs was not empty
                print_color(f"Iter {iteration}: No outputs from forward pass for candidate 'a'. Skipping.", 'yellow')
                continue

            target_for_a = batchify(*targets_from_train)
            feedback_for_a = batchify(*feedbacks_from_train).data
            score_for_a_on_train_batch = np.mean([s for s in scores_from_train if s is not None]) if any(s is not None for s in scores_from_train) else -np.inf

            self.optimizer.zero_feedback()
            self.optimizer.backward(target_for_a, feedback_for_a) # Grads for 'a' are now in optimizer

            try:
                a_prime_params_dict = self.optimizer.step(bypassing=True, verbose='output') 
                if not isinstance(a_prime_params_dict, dict) or not a_prime_params_dict:
                    print_color(f"Iter {iteration}: Optimizer.step did not return a valid param dict for a_prime. Using current agent params as a_prime.", 'yellow')
                    # Fallback: if step modified agent in-place and didn't return dict, current agent state is a_prime
                    a_prime_params_dict = {p: copy.deepcopy(p.data) for p in self.agent.parameters()}

            except Exception as e:
                print_color(f"Iter {iteration}: Error during optimizer.step for a_prime: {e}. Skipping candidate generation.", 'red')
                continue
            
            # 4. Evaluate 'a_prime' on samples of validation set
            a_prime_score, a_prime_evals = self._evaluate_candidate(
                a_prime_params_dict, train_dataset, guide, evaluation_batch_size, num_threads # Use train_dataset and guide
            )
            self._total_evaluations_tracker += a_prime_evals
            total_samples += evaluation_batch_size + train_batch_size
            metrics['new_candidate_scores'].append(a_prime_score)
            print_color(f"Iter {iteration}: New candidate a_prime generated. Validation Score: {a_prime_score:.4f}, Evals: {a_prime_evals}", 'cyan')

            # 5. Update the stats of 'a' (action_candidate_a) based on the training batch experience
            if score_for_a_on_train_batch > -np.inf:
                action_candidate_a['score_sum'] += score_for_a_on_train_batch * len(train_xs) # score is often an average
                action_candidate_a['eval_count'] += len(train_xs) # or 1 if score is total
                self._total_evaluations_tracker += len(train_xs) # training batch also counts as evaluations for UCB total T

            # 6. Add 'a_prime' (with its validation stats) to the buffer
            if a_prime_score > -np.inf and a_prime_evals > 0:
                new_candidate_entry = {
                    'params': a_prime_params_dict, 
                    'score_sum': a_prime_score * a_prime_evals, # Store sum
                    'eval_count': a_prime_evals,
                    'ucb_score': 0.0, # Will be updated
                    'iteration_created': iteration
                }
                
                # Eviction logic before adding if buffer is at max_len
                if len(self.buffer) == self.max_buffer_size:
                    self._update_buffer_ucb_scores() # Ensure UCBs are current before eviction
                    candidate_to_evict = min(self.buffer, key=lambda c: c['ucb_score'])
                    self.buffer.remove(candidate_to_evict)
                    print_color(f"Iter {iteration}: Buffer full. Evicted a candidate (UCB: {candidate_to_evict['ucb_score']:.4f})", 'magenta')
                
                self.buffer.append(new_candidate_entry)
                print_color(f"Iter {iteration}: Added new candidate to buffer.", 'magenta')
            else:
                print_color(f"Iter {iteration}: New candidate a_prime had invalid score/evals, not added to buffer.", 'yellow')

            # Update all UCB scores in the buffer after potential additions/removals/stat updates
            self._update_buffer_ucb_scores()

            # Logging
            best_in_buffer = max(self.buffer, key=lambda c: c['score_sum']/(c['eval_count'] or 1))
            metrics['best_candidate_scores'].append(best_in_buffer['score_sum']/(best_in_buffer['eval_count'] or 1))
            metrics['buffer_avg_score'].append(np.mean([c['score_sum']/(c['eval_count'] or 1) for c in self.buffer if c['eval_count'] > 0]))
            metrics['buffer_avg_evals'].append(np.mean([c['eval_count'] for c in self.buffer]))

            if iteration % log_frequency == 0:
                log_data = {
                    "iteration": iteration,
                    "best_score": metrics['best_candidate_scores'][-1], #best_candidate_score_in_buffer
                    "selected_action_ucb": action_candidate_a['ucb_score'],
                    "new_candidate_score": a_prime_score,
                    "buffer_size": len(self.buffer),
                    "buffer_avg_score": metrics['buffer_avg_score'][-1],
                    "buffer_avg_evals": metrics['buffer_avg_evals'][-1],
                    "total_evaluations_tracker": self._total_evaluations_tracker,
                    "total_samples": total_samples # Add new metric
                }
                print_color(f"Log @ Iter {iteration}: Best score in buffer: {log_data['best_score']:.4f}, Buffer size: {log_data['buffer_size']}, Total samples: {total_samples}", 'green')
            
            # Save agent (e.g., the one with highest mean score in buffer)
            if save_frequency is not None and iteration % save_frequency == 0:
                best_overall_candidate = max(self.buffer, key=lambda c: c['score_sum'] / (c['eval_count'] or 1E-9) )
                self.optimizer.update(best_overall_candidate['params']) # Load params using optimizer
                self.save_agent(save_path, iteration) # save_agent is from AlgorithmBase
                print_color(f"Iter {iteration}: Saved agent based on best candidate in buffer.", 'green')

        # End of search loop
        print_color("UCB search finished.", 'blue')
        if not self.buffer:
            print_color("Buffer is empty at the end of search. No best candidate found.", 'red')
            return metrics, -np.inf
            
        # Select the best candidate based on highest mean score (exploitation)
        final_best_candidate = max(self.buffer, key=lambda c: c['score_sum'] / (c['eval_count'] or 1E-9))
        final_best_score = final_best_candidate['score_sum'] / (final_best_candidate['eval_count'] or 1E-9)
        print_color(f"Final best candidate: Mean Score {final_best_score:.4f}, Evals {final_best_candidate['eval_count']}", 'green')

        # Load best parameters into the agent
        self.optimizer.update(final_best_candidate['params']) # Load params using optimizer

        return metrics, float(final_best_score)
    
    def select(self, buffer):
        '''Could be subclassed to implement different selection strategies'''
        return max(buffer, key=lambda c: c['ucb_score'])


class HybridUCB_LLM(MinibatchAlgorithm):
    """
    UCB Search Algorithm with Function Approximation (LLM).

    Keeps a buffer of candidates.
    In each iteration:
    - With probability alpha:
        1. Picks a candidate 'a' from the buffer with the highest UCB score.
        2. Updates the optimizer with 'a's parameters.
        3. Draws a minibatch from the training set, performs a forward/backward pass, and calls optimizer.step() to get a new candidate 'a_prime'.
        4. Evaluates 'a_prime' on a validation set minibatch.
        5. Updates statistics of 'a' (based on the training minibatch).
        6. Adds 'a_prime' (with its validation stats) to the buffer.
    - With probability 1-alpha:
        1. Uses an external LLM, prompted with candidates from the buffer, to generate a new candidate 'a_prime'.
        2. Evaluates 'a_prime' on a validation set minibatch.
        3. Adds 'a_prime' (with its validation stats) to the buffer.
    If the buffer is full, evicts the candidate with the lowest UCB score.
    """

    def __init__(self,
                 agent: trace.Module,
                 optimizer,
                 max_buffer_size: int = 10,
                 ucb_exploration_factor: float = 1.0,
                 alpha: float = 0.7,
                 llm_model: str = "vertex_ai/gemini-2.0-flash",
                 logger=None,
                 num_threads: int = None,
                 *args,
                 **kwargs):
        super().__init__(agent, optimizer, num_threads=num_threads, logger=logger, *args, **kwargs)
        
        self.alpha = alpha
        self.llm_model = llm_model
        self.llm_prompt_budget_factor = 0.5
        
        self.buffer = deque(maxlen=max_buffer_size) 
        self.max_buffer_size = max_buffer_size
        self.ucb_exploration_factor = ucb_exploration_factor

        if not hasattr(self.optimizer, 'step'):
            raise ValueError("Optimizer must have a 'step' method.")

        self._total_evaluations_tracker = 0

        # Initialize LiteLLM
        self.llm = LiteLLM(model=self.llm_model)
        print_color(f"Initialized HybridUCB_LLM with alpha={self.alpha}, LLM model={self.llm_model}", "cyan")

    def _sample_minibatch(self, dataset: Dict[str, List[Any]], batch_size: int) -> Tuple[List[Any], List[Any]]:
        """Sample a minibatch from the dataset."""
        if not dataset or not dataset.get('inputs') or not dataset.get('infos'):
            print_color("Warning: Attempted to sample from an empty or malformed dataset.", color='yellow')
            return [], []
        
        dataset_size = len(dataset['inputs'])
        if dataset_size == 0:
            print_color("Warning: Dataset is empty, cannot sample minibatch.", color='yellow')
            return [], []

        actual_batch_size = min(batch_size, dataset_size)
        indices = np.random.choice(dataset_size, actual_batch_size, replace=False)
        xs = [dataset['inputs'][i] for i in indices]
        infos = [dataset['infos'][i] for i in indices]
        return xs, infos

    def _evaluate_candidate(self, 
                              params_to_eval_dict: Dict[str, Any], 
                              dataset: Dict[str, List[Any]], 
                              guide, 
                              evaluation_batch_size: int, 
                              num_threads: Optional[int] = None
                              ) -> Tuple[float, int]:
        """Evaluates a given set of parameters on samples from the provided dataset."""
        if not dataset or not dataset.get('inputs') or not dataset.get('infos') or not dataset['inputs']:
            print_color("Evaluation dataset is empty or invalid. Returning score -inf, count 0.", color='yellow')
            return -np.inf, 0

        original_params_backup = {p: copy.deepcopy(p.data) for p in self.agent.parameters()}
        
        try:
            self.optimizer.update(params_to_eval_dict)
        except Exception as e:
            print_color(f"Error updating agent with params_to_eval_dict: {e}. Using current agent state for eval.", "red")

        eval_xs, eval_infos = self._sample_minibatch(dataset, evaluation_batch_size)
        
        if not eval_xs:
            print_color("Evaluation minibatch is empty. Returning score -inf, count 0.", color='yellow')
            self.optimizer.update(original_params_backup)
            return -np.inf, 0

        eval_scores = evaluate(self.agent,
                               guide,
                               eval_xs,
                               eval_infos,
                               min_score=self.min_score if hasattr(self, 'min_score') else None,
                               num_threads=num_threads or self.num_threads,
                               description=f"Evaluating candidate")

        self.optimizer.update(original_params_backup)

        avg_score = np.mean(eval_scores) if eval_scores and all(s is not None for s in eval_scores) else -np.inf
        eval_count = len(eval_xs) 
        
        return float(avg_score), eval_count

    def _calculate_ucb(self, candidate_buffer_entry: Dict, total_tracked_evaluations: int) -> float:
        """Calculates UCB score for a candidate in the buffer."""
        if candidate_buffer_entry['eval_count'] == 0:
            return float('inf') 
        
        mean_score = candidate_buffer_entry['score_sum'] / candidate_buffer_entry['eval_count']
        
        if total_tracked_evaluations == 0: 
             total_tracked_evaluations = 1
        
        exploration_term = self.ucb_exploration_factor * \
                           math.sqrt(math.log(total_tracked_evaluations + 1e-9) / candidate_buffer_entry['eval_count'])
        
        return mean_score + exploration_term

    def _update_buffer_ucb_scores(self):
        """Recalculates and updates UCB scores for all candidates in the buffer."""
        if not self.buffer:
            return
        
        for candidate_entry in self.buffer:
            candidate_entry['ucb_score'] = self._calculate_ucb(candidate_entry, self._total_evaluations_tracker)

    def _llm_generate_candidate(self) -> Optional[Dict[trace.nodes.ParameterNode, str]]:
        """
        Prompts an LLM with current buffer candidates to generate new string values for parameters.
        Returns a dictionary mapping ParameterNode objects to new string values, or None on failure.
        """
        print_color("Attempting to generate candidate using LLM...", "blue")
        if not self.buffer:
            print_color("LLM generation: Buffer is empty, cannot provide context to LLM.", "yellow")
            return None

        sorted_buffer = sorted(list(self.buffer), key=lambda c: c.get('ucb_score', -float('inf')), reverse=True)
        prompt_candidates = sorted_buffer

        serializable_candidate_summaries = []
        for cand_entry in prompt_candidates:
            summary = {
                "parameters":  {getattr(p,'py_name'): copy.deepcopy(p.data) for p in cand_entry['params']},
                "eval_count": cand_entry['eval_count'],
                "ucb_score": round(cand_entry.get('ucb_score',0), 4),
            }
            serializable_candidate_summaries.append(summary)
        
        example_param_structure_json_str = {getattr(p,'py_name'): copy.deepcopy(p.data) for p in self.agent.parameters()}

        prompt_messages = [
            {"role": "system", "content": "You are an expert in model optimization. Your task is to propose new string values for model parameters with high UCB scores. Please output ONLY a valid JSON dictionary where keys are parameter names and values are the new string values for those parameters, matching the example structure provided. Do not add any explanations or markdown formatting around the JSON."},
            {"role": "user", "content": f"Here are some current candidates from the search buffer and their statistics:\\n{serializable_candidate_summaries}\\n\\nHere is an example of the required JSON output structure (parameter names as keys, new string values as values):\\n{example_param_structure_json_str}\\n\\nPlease generate a new set of parameters in exactly the same JSON format. Make sure use double quotes for the keys and values."}
        ]
        
        print_color(f"LLM prompt (summary): {len(prompt_candidates)} candidates, structure example provided.", "magenta")
        
        llm_response = self.llm(prompt_messages) 
        llm_response_str = llm_response.choices[0].message.content

        if not llm_response_str:
            print_color("LLM returned an empty response.", "red")
            return None
        
        # Clean the response string
        cleaned_llm_response_str = llm_response_str.strip()
        if cleaned_llm_response_str.startswith("```json"):
            cleaned_llm_response_str = cleaned_llm_response_str[7:]
            if cleaned_llm_response_str.endswith("```"):
                cleaned_llm_response_str = cleaned_llm_response_str[:-3]
        elif cleaned_llm_response_str.startswith("```"):
                cleaned_llm_response_str = cleaned_llm_response_str[3:]
                if cleaned_llm_response_str.endswith("```"):
                    cleaned_llm_response_str = cleaned_llm_response_str[:-3]
        cleaned_llm_response_str = cleaned_llm_response_str.strip()

        if not cleaned_llm_response_str:
            print_color("LLM response was empty after cleaning markdown/whitespace.", "red")
            return None

        print_color(f"Cleaned LLM response: '{cleaned_llm_response_str}'", "magenta")
        
        # Fix common JSON formatting issues from LLM responses
        try:
            llm_params_raw = json.loads(cleaned_llm_response_str)
        except json.JSONDecodeError as e:
            print_color(f"Initial JSON parsing failed: {e}", "yellow")
            print_color("Attempting to fix JSON formatting...", "yellow")
            
            fixed_json_str = smart_quote_replacement(cleaned_llm_response_str)
            
            try:
                llm_params_raw = json.loads(fixed_json_str)
                print_color("Successfully fixed JSON formatting", "green")
            except json.JSONDecodeError as e2:
                print_color(f"Smart quote replacement failed: {e2}", "yellow")
                try:
                    simple_fixed = cleaned_llm_response_str.replace("'", '"')
                    llm_params_raw = json.loads(simple_fixed)
                    print_color("Fallback simple replacement succeeded", "green")
                except json.JSONDecodeError as e3:
                    print_color(f"All JSON parsing attempts failed: {e3}", "red")
                    print_color("Returning the candidate with the highest UCB score in the buffer.", "red")
                    return max(self.buffer, key=lambda c: c.get('ucb_score', -float('inf')))['params']

        if not isinstance(llm_params_raw, dict):
            print_color(f"LLM output was not a JSON dictionary after parsing: {type(llm_params_raw)}", "red")
            print_color("Returning the candidate with the highest UCB score in the buffer.", "red")
            return max(self.buffer, key=lambda c: c.get('ucb_score', -float('inf')))['params']

        candidate_params_dict = self.construct_update_dict(llm_params_raw)
        return candidate_params_dict
    
    def construct_update_dict(self, suggestion: Dict[str, Any]) -> Dict[ParameterNode, Any]:
        """Convert the suggestion in text into the right data type."""
        update_dict = {}
        for node in self.agent.parameters():
            if node.trainable and node.py_name in suggestion:
                try:
                    formatted_suggestion = suggestion[node.py_name]
                    if type(formatted_suggestion) == str and 'def' in formatted_suggestion:
                        formatted_suggestion = format_str(formatted_suggestion, mode=FileMode())
                    update_dict[node] = type(node.data)(formatted_suggestion)
                except (ValueError, KeyError) as e:
                    if getattr(self, 'ignore_extraction_error', False):
                        warnings.warn(
                            f"Cannot convert the suggestion '{suggestion[node.py_name]}' for {node.py_name} to the right data type"
                        )
                    else:
                        raise e
        return update_dict

    def train(self,
              guide, 
              train_dataset: Dict[str, List[Any]],
              *,
              num_search_iterations: int = 100,
              train_batch_size: int = 5, 
              evaluation_batch_size: int = 5,
              ensure_improvement: bool = False,
              improvement_threshold: float = 0.,
              eval_frequency: int = 1, 
              log_frequency: Optional[int] = None,
              save_frequency: Optional[int] = None,
              save_path: str = "checkpoints/ucb_llm_agent.pkl",
              min_score_for_agent_update: Optional[float] = None,
              verbose: Union[bool, str] = False,
              num_threads: Optional[int] = None,
              **kwargs
              ) -> Tuple[Dict[str, Any], float]:
        
        num_threads = num_threads or self.num_threads
        log_frequency = log_frequency or eval_frequency
        self.min_score = min_score_for_agent_update 
        total_samples = 0

        metrics = {
            'best_candidate_scores': [], 
            'selected_action_ucb': [],
            'new_candidate_scores': [], 
            'buffer_avg_score': [],
            'buffer_avg_evals': [],
            'llm_generation_failures': 0,
            'generation_path': []
        }

        # Initial candidate evaluation
        print_color("Evaluating initial parameters using train_dataset samples...", 'cyan')
        initial_params_dict = {p: copy.deepcopy(p.data) for p in self.agent.parameters()}
         
        initial_score, initial_evals = self._evaluate_candidate(
            initial_params_dict, train_dataset, guide, evaluation_batch_size, num_threads
        )
        self._total_evaluations_tracker += initial_evals 
        total_samples += initial_evals

        initial_candidate_entry = {
            'params': initial_params_dict,
            'score_sum': initial_score * initial_evals if initial_score > -np.inf else 0,
            'eval_count': initial_evals,
            'ucb_score': 0.0, 
            'iteration_created': 0
        }
        self.buffer.append(initial_candidate_entry)
        self._update_buffer_ucb_scores() 
        print_color(f"Initial candidate: Score {initial_score:.4f}, Evals {initial_evals}", 'yellow')
        
        # Main search loop
        for iteration in range(1, num_search_iterations + 1):
            if not self.buffer:
                print_color("Buffer is empty, stopping search.", 'red')
                break

            self._update_buffer_ucb_scores()
            a_prime_params_dict = None
            a_prime_score = -np.inf
            a_prime_evals = 0
            generation_method = "none"

            if random.random() < self.alpha: # UCB Path
                generation_method = "ucb"
                metrics['generation_path'].append("ucb")
                if not self.buffer:
                    print_color(f"Iter {iteration} (UCB Path): Buffer empty, cannot select action. Skipping.", "red")
                    continue

                action_candidate_a = self.select(self.buffer)
                
                selected_mean_score = action_candidate_a['score_sum'] / action_candidate_a['eval_count'] if action_candidate_a['eval_count'] > 0 else -np.inf
                print_color(f"Iter {iteration} (UCB Path): Selected action candidate (UCB: {action_candidate_a['ucb_score']:.4f}, MeanScore: {selected_mean_score:.4f} Evals: {action_candidate_a['eval_count']})", 'blue')
                metrics['selected_action_ucb'].append(action_candidate_a['ucb_score'])

                self.optimizer.update(action_candidate_a['params'])

                train_xs, train_infos = self._sample_minibatch(train_dataset, train_batch_size)
                if not train_xs:
                    print_color(f"Iter {iteration} (UCB Path): Training minibatch empty, skipping optimizer step.", 'yellow')
                    continue 
                
                total_samples += len(train_xs)

                # Forward pass for 'a'
                outputs_for_a = []
                use_asyncio = self._use_asyncio(num_threads)
                if use_asyncio:
                    outputs_for_a = async_run([self.forward]*len(train_xs),
                                       [(self.agent, x, guide, info) for x, info in zip(train_xs, train_infos)],
                                       max_workers=num_threads,
                                       description=f"Iter {iteration} (UCB): Forward for 'a'")
                else:
                    outputs_for_a = [self.forward(self.agent, x, guide, info) for x, info in zip(train_xs, train_infos)]

                scores_from_train, targets_from_train, feedbacks_from_train = [], [], []
                for target, score, feedback in outputs_for_a:
                    scores_from_train.append(score)
                    targets_from_train.append(target)
                    feedbacks_from_train.append(feedback)
                
                if not scores_from_train:
                    print_color(f"Iter {iteration} (UCB Path): No outputs from forward pass for 'a'. Skipping.", 'yellow')
                    continue

                target_for_a = batchify(*targets_from_train)
                feedback_for_a = batchify(*feedbacks_from_train).data
                score_for_a_on_train_batch = np.mean([s for s in scores_from_train if s is not None]) if any(s is not None for s in scores_from_train) else -np.inf

                self.optimizer.zero_feedback()
                self.optimizer.backward(target_for_a, feedback_for_a)

                # Get a_prime by optimizer step
                try:
                    returned_params = self.optimizer.step(bypassing=True, verbose=(verbose if isinstance(verbose, str) else 'output')) 
                    if not isinstance(returned_params, dict) or not returned_params:
                        print_color(f"Iter {iteration} (UCB Path): Optimizer.step did not return a valid param dict for a_prime. Using current agent params.", 'yellow')
                        a_prime_params_dict = {p: copy.deepcopy(p.data) for p in self.agent.parameters()}
                    else:
                        a_prime_params_dict = {p: copy.deepcopy(p.data)  for p in returned_params}

                except Exception as e:
                    print_color(f"Iter {iteration} (UCB Path): Error during optimizer.step for a_prime: {e}. Skipping.", 'red')
                    continue
                
                # Evaluate a_prime (from UCB path)
                a_prime_score, a_prime_evals = self._evaluate_candidate(
                    a_prime_params_dict, train_dataset, guide, evaluation_batch_size, num_threads
                )
                self._total_evaluations_tracker += a_prime_evals
                total_samples += a_prime_evals

                # Update stats of action_candidate_a
                if score_for_a_on_train_batch > -np.inf:
                    action_candidate_a['score_sum'] += score_for_a_on_train_batch * len(train_xs)
                    action_candidate_a['eval_count'] += len(train_xs)
                    self._total_evaluations_tracker += len(train_xs)
                
                print_color(f"Iter {iteration} (UCB Path): New candidate a_prime (from UCB) generated. Eval Score: {a_prime_score:.4f}, Evals: {a_prime_evals}", 'cyan')

            else: # LLM Path
                generation_method = "llm"
                metrics['generation_path'].append("llm")
                print_color(f"Iter {iteration} (LLM Path): Generating candidate via LLM.", 'blue')
                a_prime_params_dict = self._llm_generate_candidate()

                if a_prime_params_dict:
                    # Evaluate a_prime (from LLM path)
                    a_prime_score, a_prime_evals = self._evaluate_candidate(
                        a_prime_params_dict, train_dataset, guide, evaluation_batch_size, num_threads
                    )
                    self._total_evaluations_tracker += a_prime_evals
                    total_samples += a_prime_evals
                    print_color(f"Iter {iteration} (LLM Path): New candidate a_prime (from LLM) generated. Eval Score: {a_prime_score:.4f}, Evals: {a_prime_evals}", 'cyan')
                else:
                    print_color(f"Iter {iteration} (LLM Path): LLM failed to generate a valid candidate. Skipping addition to buffer.", 'red')
                    metrics['llm_generation_failures'] += 1
                    continue

            # Common logic for adding a_prime to buffer
            metrics['new_candidate_scores'].append(a_prime_score)

            if a_prime_params_dict and a_prime_score > -np.inf and a_prime_evals > 0:
                new_candidate_entry = {
                    'params': a_prime_params_dict,
                    'score_sum': a_prime_score * a_prime_evals,
                    'eval_count': a_prime_evals,
                    'ucb_score': 0.0, 
                    'iteration_created': iteration
                }
                
                if len(self.buffer) == self.max_buffer_size:
                    self._update_buffer_ucb_scores()
                    candidate_to_evict = min(self.buffer, key=lambda c: c['ucb_score'])
                    self.buffer.remove(candidate_to_evict)
                    evicted_mean_score = candidate_to_evict['score_sum'] / candidate_to_evict['eval_count'] if candidate_to_evict['eval_count'] > 0 else -np.inf
                    print_color(f"Iter {iteration}: Buffer full. Evicted candidate (UCB: {candidate_to_evict['ucb_score']:.4f}, MeanScore: {evicted_mean_score:.4f})", 'magenta')
                
                self.buffer.append(new_candidate_entry)
                print_color(f"Iter {iteration}: Added new candidate (from {generation_method}) to buffer.", 'magenta')
            elif a_prime_params_dict:
                print_color(f"Iter {iteration}: New candidate a_prime (from {generation_method}) had invalid score/evals ({a_prime_score}, {a_prime_evals}), not added to buffer.", 'yellow')

            self._update_buffer_ucb_scores()

            # Logging
            if self.buffer:
                best_in_buffer = max(self.buffer, key=lambda c: (c['score_sum']/(c['eval_count'] if c['eval_count'] > 0 else 1)))
                current_best_score = best_in_buffer['score_sum']/(best_in_buffer['eval_count'] if best_in_buffer['eval_count'] > 0 else 1)
                metrics['best_candidate_scores'].append(current_best_score)
                
                valid_scores = [c['score_sum']/(c['eval_count'] if c['eval_count'] > 0 else 1) for c in self.buffer if c['eval_count'] > 0]
                metrics['buffer_avg_score'].append(np.mean(valid_scores) if valid_scores else -np.inf)
                metrics['buffer_avg_evals'].append(np.mean([c['eval_count'] for c in self.buffer]))
            else:
                metrics['best_candidate_scores'].append(-np.inf)
                metrics['buffer_avg_score'].append(-np.inf)
                metrics['buffer_avg_evals'].append(0)

            if iteration % log_frequency == 0:
                log_data = {
                    "iteration": iteration,
                    "best_score": metrics['best_candidate_scores'][-1],
                    "newly_evaluated_candidate_score": a_prime_score,
                    "buffer_size": len(self.buffer),
                    "buffer_avg_score": metrics['buffer_avg_score'][-1],
                    "buffer_avg_evals": metrics['buffer_avg_evals'][-1],
                    "total_evaluations_ucb_T": self._total_evaluations_tracker,
                    "total_samples": total_samples,
                    "generation_method_this_iter": generation_method,
                    "llm_generation_total_failures": metrics['llm_generation_failures']
                }
                if generation_method == "ucb" and metrics['selected_action_ucb']:
                    log_data["selected_action_ucb"] = metrics['selected_action_ucb'][-1]
                
                print_color(f"Log @ Iter {iteration}: Best score in buffer: {log_data['best_score']:.4f}, Gen method: {generation_method}, Buffer size: {len(self.buffer)}, Total samples: {total_samples}", 'green')
            
            if save_frequency is not None and iteration % save_frequency == 0 and self.buffer:
                best_overall_candidate_entry = max(self.buffer, key=lambda c: (c['score_sum'] / (c['eval_count'] if c['eval_count'] > 0 else 1E-9)))
                self.optimizer.update(best_overall_candidate_entry['params']) 
                if hasattr(self, 'save_agent'):
                    self.save_agent(save_path, iteration) 
                    best_mean_score_for_save = best_overall_candidate_entry['score_sum'] / (best_overall_candidate_entry['eval_count'] if best_overall_candidate_entry['eval_count'] > 0 else 1E-9)
                    print_color(f"Iter {iteration}: Saved agent based on best candidate in buffer (Mean Score: {best_mean_score_for_save:.4f}).", 'green')
                else:
                    print_color(f"Iter {iteration}: save_agent method not found, skipping save.", 'yellow')

        print_color("UCB-LLM search finished.", 'blue')
        if not self.buffer:
            print_color("Buffer is empty at the end of search. No best candidate found.", 'red')
            return metrics, -np.inf
            
        final_best_candidate = max(self.buffer, key=lambda c: (c['score_sum'] / (c['eval_count'] if c['eval_count'] > 0 else 1E-9)))
        final_best_score = final_best_candidate['score_sum'] / (final_best_candidate['eval_count'] if final_best_candidate['eval_count'] > 0 else 1E-9)
        final_best_evals = final_best_candidate['eval_count']
        print_color(f"Final best candidate: Mean Score {final_best_score:.4f}, Evals {final_best_evals}", 'green')

        self.optimizer.update(final_best_candidate['params'])

        return metrics, float(final_best_score)
    
    def select(self, buffer):
        '''Selects candidate with highest UCB score.'''
        if not buffer: return None
        return max(buffer, key=lambda c: c.get('ucb_score', -float('inf')))


class UCBSearchFunctionApproximationAlgorithm(UCBSearchAlgorithm):
    """
    UCB Search Algorithm that uses LLM function approximation to select candidates.
    """
    
    def __init__(self, llm_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm_model = llm_model
        self.llm = LiteLLM(model=self.llm_model)
        print_color(f"Initialized UCBSearchFunctionApproximationAlgorithm with LLM model={self.llm_model}", "cyan")
    
    def select(self, buffer): 
        """Generate a new candidate entry using LLM. Note: this doesn't add it to the buffer."""
        new_action_params = self._llm_generate_candidate()
        new_candidate_entry = {
            'params': new_action_params,
            'score_sum': 0,
            'eval_count': 0,
            'ucb_score': 0.0, 
            'iteration_created': 0
        }
        return new_candidate_entry
    
    def _llm_generate_candidate(self) -> Optional[Dict[trace.nodes.ParameterNode, str]]:
        """
        Prompts an LLM with current buffer candidates to generate new string values for parameters.
        Returns a dictionary mapping ParameterNode objects to new string values, or None on failure.
        """
        print_color("Attempting to generate candidate using LLM...", "blue")
        if not self.buffer:
            print_color("LLM generation: Buffer is empty, cannot provide context to LLM.", "yellow")
            return None

        sorted_buffer = sorted(list(self.buffer), key=lambda c: c.get('ucb_score', -float('inf')), reverse=True)
        prompt_candidates = sorted_buffer

        serializable_candidate_summaries = []
        for cand_entry in prompt_candidates:
            summary = {
                "parameters":  {getattr(p,'py_name'): copy.deepcopy(p.data) for p in cand_entry['params']},
                "eval_count": cand_entry['eval_count'],
                "ucb_score": round(cand_entry.get('ucb_score',0), 4),
            }
            serializable_candidate_summaries.append(summary)
        
        example_param_structure_json_str = {getattr(p,'py_name'): copy.deepcopy(p.data) for p in self.agent.parameters()}

        prompt_messages = [
            {"role": "system", "content": "You are an expert in model optimization. Your task is to propose new string values for model parameters with high UCB scores. Please output ONLY a valid JSON dictionary where keys are parameter names and values are the new string values for those parameters, matching the example structure provided. Do not add any explanations or markdown formatting around the JSON."},
            {"role": "user", "content": f"Here are some current candidates from the search buffer and their statistics:\\n{serializable_candidate_summaries}\\n\\nHere is an example of the required JSON output structure (parameter names as keys, new string values as values):\\n{example_param_structure_json_str}\\n\\nPlease generate a new set of parameters in exactly the same JSON format. Make sure use double quotes for the keys and values."}
        ]
        
        print_color(f"LLM prompt (summary): {len(prompt_candidates)} candidates, structure example provided.", "magenta")
        
        llm_response = self.llm(prompt_messages) 
        llm_response_str = llm_response.choices[0].message.content

        if not llm_response_str:
            print_color("LLM returned an empty response.", "red")
            return None
        
        # Clean the response string
        cleaned_llm_response_str = llm_response_str.strip()
        if cleaned_llm_response_str.startswith("```json"):
            cleaned_llm_response_str = cleaned_llm_response_str[7:]
            if cleaned_llm_response_str.endswith("```"):
                cleaned_llm_response_str = cleaned_llm_response_str[:-3]
        elif cleaned_llm_response_str.startswith("```"):
                cleaned_llm_response_str = cleaned_llm_response_str[3:]
                if cleaned_llm_response_str.endswith("```"):
                    cleaned_llm_response_str = cleaned_llm_response_str[:-3]
        cleaned_llm_response_str = cleaned_llm_response_str.strip()

        if not cleaned_llm_response_str:
            print_color("LLM response was empty after cleaning markdown/whitespace.", "red")
            return None

        print_color(f"Cleaned LLM response: '{cleaned_llm_response_str}'", "magenta")
        
        # Fix common JSON formatting issues from LLM responses
        try:
            llm_params_raw = json.loads(cleaned_llm_response_str)
        except json.JSONDecodeError as e:
            print_color(f"Initial JSON parsing failed: {e}", "yellow")
            print_color("Attempting to fix JSON formatting...", "yellow")
            
            fixed_json_str = smart_quote_replacement(cleaned_llm_response_str)
            
            try:
                llm_params_raw = json.loads(fixed_json_str)
                print_color("Successfully fixed JSON formatting", "green")
            except json.JSONDecodeError as e2:
                print_color(f"Smart quote replacement failed: {e2}", "yellow")
                try:
                    simple_fixed = cleaned_llm_response_str.replace("'", '"')
                    llm_params_raw = json.loads(simple_fixed)
                    print_color("Fallback simple replacement succeeded", "green")
                except json.JSONDecodeError as e3:
                    print_color(f"All JSON parsing attempts failed: {e3}", "red")
                    print_color("Returning the candidate with the highest UCB score in the buffer.", "red")
                    return max(self.buffer, key=lambda c: c.get('ucb_score', -float('inf')))['params']

        if not isinstance(llm_params_raw, dict):
            print_color(f"LLM output was not a JSON dictionary after parsing: {type(llm_params_raw)}", "red")
            print_color("Returning the candidate with the highest UCB score in the buffer.", "red")
            return max(self.buffer, key=lambda c: c.get('ucb_score', -float('inf')))['params']

        candidate_params_dict = self.construct_update_dict(llm_params_raw)
        return candidate_params_dict
    
    def construct_update_dict(self, suggestion: Dict[str, Any]) -> Dict[ParameterNode, Any]:
        """Convert the suggestion in text into the right data type."""
        update_dict = {}
        for node in self.agent.parameters():
            if node.trainable and node.py_name in suggestion:
                try:
                    formatted_suggestion = suggestion[node.py_name]
                    if type(formatted_suggestion) == str and 'def' in formatted_suggestion:
                        formatted_suggestion = format_str(formatted_suggestion, mode=FileMode())
                    update_dict[node] = type(node.data)(formatted_suggestion)
                except (ValueError, KeyError) as e:
                    if getattr(self, 'ignore_extraction_error', False):
                        warnings.warn(
                            f"Cannot convert the suggestion '{suggestion[node.py_name]}' for {node.py_name} to the right data type"
                        )
                    else:
                        raise e
        return update_dict

