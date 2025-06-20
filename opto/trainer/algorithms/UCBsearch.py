import numpy as np
import copy
import math
from collections import deque
from typing import Union, List, Tuple, Dict, Any, Optional
from opto import trace
from opto.trainer.utils import async_run # Assuming print_color is in utils
from opto.optimizers.utils import print_color
from opto.trainer.algorithms.basic_algorithms import MinibatchAlgorithm, evaluate, batchify # evaluate and batchify might be useful

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
                 ucb_exploration_factor: float = 1.0,  # Controls exploration vs exploitation tradeoff in UCB selection
                                                     # UCB formula: μ(a) + c * sqrt(ln(t) / n(a)), c is the exploration factor
                 logger=None,
                 num_threads: int = None,
                 *args,
                 **kwargs):
        super().__init__(agent, optimizer, num_threads=num_threads, logger=logger, *args, **kwargs)
        
        self.buffer = deque(maxlen=max_buffer_size) 
        self.max_buffer_size = max_buffer_size
        # UCB exploration factor: Higher values encourage more exploration of less-tested candidates,
        # lower values favor exploitation of well-performing candidates. 
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

        original_params = {p: copy.deepcopy(p.data) for p in self.optimizer.parameters}
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
        
        # UCB exploration term: ucb_exploration_factor scales the confidence interval
        # Higher factor = more exploration, lower factor = more exploitation
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
              validation_dataset: Optional[Dict[str, List[Any]]] = None,  # Validation set for evaluation, defaults to train_dataset
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
        # Default validation_dataset to train_dataset if not provided
        if validation_dataset is None:
            validation_dataset = train_dataset
            
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
        print_color("Evaluating initial parameters using validation_dataset samples...", 'cyan')
        initial_params_dict = {p: copy.deepcopy(p.data) for p in self.optimizer.parameters}
        initial_score, initial_evals = self._evaluate_candidate(
            initial_params_dict, validation_dataset, guide, evaluation_batch_size, num_threads # Use validation_dataset and guide
        )
        self._total_evaluations_tracker += initial_evals 
        total_samples += initial_evals

        # Log initial evaluation
        self.logger.log('Initial UCB score', initial_score, 0, color='blue')
        self.logger.log('Initial evaluations', initial_evals, 0, color='cyan')

        initial_candidate_entry = {
            'params': initial_params_dict,
            'score_sum': initial_score * initial_evals if initial_score > -np.inf else 0, # Store sum for accurate mean later
            'eval_count': initial_evals,
            'ucb_score': None, # avoid accidental reads before it's initialized
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
            
            # Log selected action UCB score
            self.logger.log('Selected action UCB', action_candidate_a['ucb_score'], iteration, color='magenta')
            self.logger.log('Selected action mean score', action_candidate_a['score_sum']/(action_candidate_a['eval_count'] or 1), iteration, color='cyan')
            
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
                    a_prime_params_dict = {p: copy.deepcopy(p.data) for p in self.optimizer.parameters}

            except Exception as e:
                print_color(f"Iter {iteration}: Error during optimizer.step for a_prime: {e}. Skipping candidate generation.", 'red')
                continue
            
            # 4. Evaluate 'a_prime' on samples of validation set
            a_prime_score, a_prime_evals = self._evaluate_candidate(
                a_prime_params_dict, validation_dataset, guide, evaluation_batch_size, num_threads # Use validation_dataset and guide
            )
            self._total_evaluations_tracker += a_prime_evals
            total_samples += evaluation_batch_size + train_batch_size
            metrics['new_candidate_scores'].append(a_prime_score)
            
            # Log new candidate performance
            self.logger.log('New candidate score', a_prime_score, iteration, color='green')
            self.logger.log('Training batch score', score_for_a_on_train_batch, iteration, color='yellow')
            
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
                    'ucb_score': None, # avoid accidental reads before it's initializad
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
                
                # Log all important metrics
                self.logger.log('Best candidate score', log_data['best_score'], iteration, color='green')
                self.logger.log('Buffer size', log_data['buffer_size'], iteration, color='blue')
                self.logger.log('Buffer average score', log_data['buffer_avg_score'], iteration, color='cyan')
                self.logger.log('Buffer average evaluations', log_data['buffer_avg_evals'], iteration, color='orange')
                self.logger.log('Total evaluations tracker', log_data['total_evaluations_tracker'], iteration, color='magenta')
                self.logger.log('Total samples processed', log_data['total_samples'], iteration, color='yellow')
                
                print_color(f"Log @ Iter {iteration}: Best score in buffer: {log_data['best_score']:.4f}, Buffer size: {log_data['buffer_size']}, Total samples: {total_samples}", 'green')
            
            # Save agent (e.g., the one with highest mean score in buffer)
            if save_frequency is not None and iteration % save_frequency == 0:
                best_overall_candidate = max(self.buffer, key=lambda c: c['score_sum'] / (c['eval_count'] or 1E-9) )
                self.optimizer.update(best_overall_candidate['params']) # Load params using optimizer
                self.save_agent(save_path, iteration) # save_agent is from AlgorithmBase
                print_color(f"Iter {iteration}: Saved agent based on best candidate in buffer.", 'green')

        # End of search loop
        print_color("UCB search finished.", 'blue')
        
        # Log final training summary
        final_iteration = num_search_iterations
        self.logger.log('UCB search completed', final_iteration, final_iteration, color='blue')
        self.logger.log('Final total samples', total_samples, final_iteration, color='magenta')
        
        if not self.buffer:
            print_color("Buffer is empty at the end of search. No best candidate found.", 'red')
            self.logger.log('Final status', 'Buffer empty - no best candidate', final_iteration, color='red')
            return metrics, -np.inf
            
        # Select the best candidate based on highest mean score (exploitation)
        final_best_candidate = max(self.buffer, key=lambda c: c['score_sum'] / (c['eval_count'] or 1E-9))
        final_best_score = final_best_candidate['score_sum'] / (final_best_candidate['eval_count'] or 1E-9)
        
        # Log final results
        self.logger.log('Final best score', final_best_score, final_iteration, color='green')
        self.logger.log('Final best candidate evaluations', final_best_candidate['eval_count'], final_iteration, color='cyan')
        self.logger.log('Final buffer size', len(self.buffer), final_iteration, color='blue')
        
        print_color(f"Final best candidate: Mean Score {final_best_score:.4f}, Evals {final_best_candidate['eval_count']}", 'green')

        # Load best parameters into the agent
        self.optimizer.update(final_best_candidate['params']) # Load params using optimizer

        return metrics, float(final_best_score)
    
    def select(self, buffer):
        '''Could be subclassed to implement different selection strategies'''
        return max(buffer, key=lambda c: c['ucb_score'])