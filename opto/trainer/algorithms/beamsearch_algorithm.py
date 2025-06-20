import numpy as np
import copy
from typing import Union, List, Tuple, Dict, Any, Optional
from opto.trainer.utils import async_run
from opto.optimizers.utils import print_color
from opto.trainer.algorithms.basic_algorithms import MinibatchAlgorithm, evaluate, batchify


class BeamsearchAlgorithm(MinibatchAlgorithm):
    """
    BeamsearchAlgorithm performs beam search over parameter space.
    """  """
    It starts with an initial prompt, generates multiple candidates,
    selects top beam_width candidates, and repeats this process up to max_depth.
    At each step, it evaluates candidates on a validation set to select the best ones.
    Finally output the best candidate based on validation scores.
    """

    def train(self,
              guide,
              train_dataset,
              *,
              validate_dataset=None,  # dataset for selecting the best candidates
              validate_guide=None,    # guide for validation
              validation_dataset_size=5,  # size of validation minibatch for each evaluation
              beam_width=3,           # number of candidates to keep at each beam step
              num_proposals=4,        # number of proposals to generate per beam
              max_depth=2,            # maximum depth of beam search
              num_epochs=1,
              batch_size=1,
              test_dataset=None,
              log_frequency=None,
              save_frequency=None,
              save_path="checkpoints/agent.pkl",
              min_score=None,
              num_threads=10,
              test_frequency=4,       # How often to evaluate on test set
              **kwargs
              ):
        """
        Performs beam search to find optimal parameters.
        
        Args:
            beam_width: Number of candidates to keep at each level of the beam search
            num_proposals: Number of proposals to generate per beam candidate
            max_depth: Maximum depth of the beam search
            validate_dataset: Dataset used to select the best candidates
            validate_guide: Guide used for validation
            validation_dataset_size: Size of validation minibatch for each evaluation (if None, uses all)
            test_frequency: How often to evaluate on test set (every N steps)
            Other parameters are the same as MinibatchAlgorithm.train()
        """
        self.total_samples = 0

        print_color(f"Running BeamsearchAlgorithm with beam_width={beam_width}, max_depth={max_depth}", 'blue')
        
        # Use train dataset for validation if not specified
        validate_dataset = validate_dataset or train_dataset
        validate_guide = validate_guide or guide
        self.min_score = min_score
        
        # Default validation dataset size
        if validation_dataset_size is None:
            # Use a reasonable default - e.g., 10 samples or all if dataset is smaller
            validation_dataset_size = min(10, len(validate_dataset['inputs']))
            
        print_color(f"Using validation_dataset_size={validation_dataset_size} for intermediate evaluations", 'blue')
        
        # Store original parameters to restore after each exploration
        original_params = {p: copy.deepcopy(p.data) for p in self.optimizer.parameters}
        
        # Dictionary to track metrics during beam search
        metrics = {
            'best_validation_scores': [],  # Best validation score at each depth
            'depth_scores': [],            # All scores at each depth
            'test_scores': [],             # Test scores at periodic intervals
            'test_depths': []              # Depths at which test scores were recorded
        }
        
        # Evaluate initial parameters on test set
        if test_dataset is not None:
            print_color("\n===== Evaluating Initial Parameters =====", 'blue')
            initial_test_scores = evaluate(
                self.agent,
                guide,
                test_dataset['inputs'],
                test_dataset['infos'],
                min_score=min_score,
                num_threads=num_threads,
                description="Evaluating initial parameters on test set"
            )
            initial_test_score = np.mean(initial_test_scores) if all([s is not None for s in initial_test_scores]) else -np.inf
            print_color(f"Initial test score: {initial_test_score:.4f}", 'yellow')
            
            # Log initial test score
            self.logger.log('Initial test score', initial_test_score, 0, color='blue')
            
            # Add initial score to metrics for logging
            metrics['test_scores'].append(initial_test_score)
            metrics['test_depths'].append(1) # Represent initial score at depth 0
        
        # Start with a single beam (the original parameters)
        beams = [original_params]
        
        # Run beam search for max_depth iterations
        for depth in range(max_depth):
            print_color(f"\n===== Beam Search Depth {depth+1}/{max_depth} with {len(beams)} beams =====", 'blue')
            
            # Sample a validation minibatch for this depth
            validation_xs, validation_infos = self._sample_minibatch(
                validate_dataset, 
                validation_dataset_size
            )
            
            # Create a validation mini-dataset for this depth
            validation_mini_dataset = {
                'inputs': validation_xs,
                'infos': validation_infos
            }
            
            print_color(f"Sampled validation minibatch of size {len(validation_xs)} for depth {depth+1}", 'cyan')
            
            # Collect all expanded candidates
            all_candidates = []
            
            # Process each beam in the current set
            for beam_idx, beam_params in enumerate(beams):
                print_color(f"Processing beam {beam_idx+1}/{len(beams)}", 'yellow')
                
                # Expand: Generate multiple proposals from this beam (without evaluation)
                beam_candidates = self.expand(
                    beam_params=beam_params,
                    beam_idx=beam_idx,
                    guide=guide,
                    train_dataset=train_dataset,
                    batch_size=batch_size,
                    num_proposals=num_proposals,
                    num_threads=num_threads
                )
                
                # Add all candidates to the pool for selection
                all_candidates.extend(beam_candidates)
                self.total_samples += batch_size
            # Select: Evaluate all candidates and choose the top beam_width
            beams, scores = self.select(
                candidates=all_candidates,
                validate_guide=validate_guide,
                validation_mini_dataset=validation_mini_dataset,
                beam_width=beam_width,
                num_threads=num_threads,
                min_score=min_score,
                return_scores=True  # Modified to return scores as well
            )
            self.total_samples += validation_dataset_size*len(all_candidates)
            # Track validation scores for this depth
            if len(scores) > 0:
                best_score = max(scores)
                best_idx = scores.index(best_score)
                best_params = beams[best_idx]
                metrics['best_validation_scores'].append(best_score)
                metrics['depth_scores'].append(scores)
                
                print_color(f"Depth {depth+1} - Best validation score: {best_score:.4f}", 'green')
                
                # Log validation metrics
                step_num = depth + 1
                self.logger.log('Best validation score', best_score, step_num, color='green')
                self.logger.log('Average validation score', np.mean(scores), step_num, color='cyan')
                self.logger.log('Min validation score', min(scores), step_num, color='yellow')
                self.logger.log('Max validation score', max(scores), step_num, color='magenta')
                
                # Evaluate on test set every test_frequency steps
                if test_dataset is not None and ((depth + 1) % test_frequency == 0):
                    # Update agent with best parameters from this depth
                    self.optimizer.update(best_params)
                    # Print best parameters
                    print_color("\nBest parameters at depth {}:".format(depth + 1), 'cyan')
                    for key, value in best_params.items():
                        # Try to get a clean string name from the key, which might be a parameter object
                        if hasattr(key, 'name'):
                            # Extract string name from parameter object
                            param_name = key.name
                        else:
                            # If it's already a string or doesn't have a name attribute, use it directly
                            param_name = str(key)
                        print_color(f"{param_name}: {value}", 'cyan')
                    print_color("", 'cyan')  # Empty line for readability
                    # Evaluate on test set
                    test_scores = evaluate(
                        self.agent,
                        guide,
                        test_dataset['inputs'],
                        test_dataset['infos'],
                        min_score=min_score,
                        num_threads=num_threads,
                        description=f"Evaluating best parameters at depth {depth+1} on test set"
                    )
                    test_score = np.mean(test_scores) if all([s is not None for s in test_scores]) else -np.inf
                    
                    # Record the test score
                    metrics['test_scores'].append(test_score)
                    metrics['test_depths'].append(depth + 1)
                    
                    print_color(f"Depth {depth+1} - Test score: {test_score:.4f}", 'magenta')
                    
                    # Log test score
                    self.logger.log('Periodic test score', test_score, step_num, color='magenta')
        
        # Final selection - choose the best beam using FULL validation set
        print_color("\n===== Final Selection Using Full Validation Set =====", 'blue')
        
        # Use select method with the full validation dataset
        full_validation_dataset = {
            'inputs': validate_dataset['inputs'],
            'infos': validate_dataset['infos']
        }
        
        # Select the single best beam from the final candidates
        best_beams, final_val_scores = self.select(
            candidates=beams,
            validate_guide=validate_guide,
            validation_mini_dataset=full_validation_dataset,
            beam_width=1,  # Only select the best one
            num_threads=num_threads,
            min_score=min_score,
            return_scores=True  # Return scores too
        )
        
        # Get the best parameters
        best_params = best_beams[0]
        final_validation_score = final_val_scores[0] if final_val_scores else -np.inf
        
        # Log final validation score
        final_step = max_depth + 1
        self.logger.log('Final validation score', final_validation_score, final_step, color='blue')
        
        # Apply the best parameters
        self.optimizer.update(best_params)
        
        # Print out the final proposal candidate parameters
        print_color("\n===== Final Proposal Candidate Parameters =====", 'magenta')
        for param in self.agent.parameters():
            # Use a try-except block to handle parameter lookup
            try:
                # Check if parameter object is directly available as a key
                if param in best_params:
                    param_value = best_params[param]
                # Try to find by name if available
                elif hasattr(param, 'name') and param.name in best_params:
                    param_value = best_params[param.name]
                else:
                    param_value = "Parameter not found in best_params"
                
                # Get the parameter name directly
                param_name = param.name if hasattr(param, 'name') else str(param)
                print_color(f"{param_name}: {param_value}", 'blue')
            except Exception as e:
                print_color(f"Error accessing parameter {getattr(param, 'name', str(param))}: {e}", 'red')
                continue
        
        # Evaluate on test set for reporting (if provided)
        if test_dataset is not None:
            final_test_scores = evaluate(
                self.agent,
                guide,
                test_dataset['inputs'],
                test_dataset['infos'],
                min_score=min_score,
                num_threads=num_threads,
                description="Evaluating best beam on test set"
            )
            final_test_score = np.mean(final_test_scores) if all([s is not None for s in final_test_scores]) else -np.inf
        else:
            final_test_score = None
            
        if final_test_score is not None:
            print_color(f"BEST BEAM - Test score: {final_test_score:.4f}", 'green')
            
            # Log final test score
            self.logger.log('Final test score', final_test_score, final_step, color='green')
        
        # Save the best model
        if save_frequency is not None and save_frequency > 0:
            self.save_agent(save_path, 0)
        
        # Print periodic test scores summary if available
        if metrics['test_scores']:
            print_color("\n===== Periodic Test Scores Summary =====", 'blue')
            for depth, score in zip(metrics['test_depths'], metrics['test_scores']):
                print_color(f"Depth {depth}: Test score = {score:.4f}", 'cyan')
            
        # For API consistency with other algorithms
        return metrics, final_test_score if final_test_score is not None else 0.0
    
    def _sample_minibatch(self, dataset, batch_size):
        """Sample a minibatch from the dataset."""
        indices = np.random.choice(len(dataset['inputs']), min(batch_size, len(dataset['inputs'])), replace=False)
        xs = [dataset['inputs'][i] for i in indices]
        infos = [dataset['infos'][i] for i in indices]
        return xs, infos
    
    def expand(self, 
               beam_params: Dict,
               beam_idx: int,
               guide,
               train_dataset,
               batch_size: int,
               num_proposals: int,
               num_threads: int = None) -> List[Dict]:
        """
        Expands a single candidate into multiple proposals without evaluation.
        
        Args:
            beam_params: Parameters of the current beam
            beam_idx: Index of the current beam
            guide: Guide for generating feedback
            train_dataset: Training dataset
            batch_size: Training batch size
            num_proposals: Number of proposals to generate
            num_threads: Number of threads to use
            
        Returns:
            List of parameter dictionaries for each candidate
        """
        # Restore parameters for this beam
        self.optimizer.update(beam_params)
        
        # Run forward pass on minibatch to get outputs and feedbacks
        xs_batch, infos_batch = self._sample_minibatch(train_dataset, batch_size)
        
        # Forward the agent on the minibatch
        use_asyncio = self._use_asyncio(num_threads)
        if use_asyncio:
            outputs = async_run([self.forward]*len(xs_batch),
                               [(self.agent, x, guide, info) for x, info in zip(xs_batch, infos_batch)],
                               max_workers=num_threads,
                               description=f"Forward pass (beam {beam_idx+1}, batch size: {len(xs_batch)})")
        else:
            outputs = [self.forward(self.agent, x, guide, info) for x, info in zip(xs_batch, infos_batch)]
        
        # Prepare for optimizer backward and step
        scores, targets, feedbacks = [], [], []
        for target, score, feedback in outputs:
            scores.append(score)
            targets.append(target)
            feedbacks.append(feedback)
        target = batchify(*targets)
        feedback = batchify(*feedbacks).data
        
        # Backward pass to compute gradients
        self.optimizer.zero_feedback()
        self.optimizer.backward(target, feedback)
        
        # Generate multiple proposals
        step_kwargs = dict(bypassing=True, verbose='output')
        candidates = []
        
        # Generate num_proposals candidates
        if use_asyncio:
            update_dicts = async_run([self.optimizer.step]*num_proposals,
                                    kwargs_list=[step_kwargs] * num_proposals,
                                    max_workers=num_threads,
                                    description=f"Generating {num_proposals} proposals for beam {beam_idx+1}")
        else:
            update_dicts = [self.optimizer.step(**step_kwargs) for _ in range(num_proposals)]
        
        # Collect all valid proposals
        for update_dict in update_dicts:
            if len(update_dict) > 0:
                # Make sure update_dict contains all parameters from beam_params
                # Add any missing parameters from beam_params to update_dict
                for param_key, param_value in beam_params.items():
                    if param_key not in update_dict:
                        update_dict[param_key] = param_value
                candidates.append(update_dict)
        
        # Also include the original beam parameters as a candidate
        candidates.append(beam_params)
        
        return candidates

    def select(self, 
               candidates: List[Dict],
               validate_guide,
               validation_mini_dataset,
               beam_width: int,
               num_threads: int = None,
               min_score: float = None,
               return_scores: bool = False) -> Union[List[Dict], Tuple[List[Dict], List[float]]]:
        """
        Evaluates all candidates and selects the top beam_width candidates based on validation scores.
        
        Args:
            candidates: List of parameter dictionaries for each candidate
            validate_guide: Guide for validation
            validation_mini_dataset: Validation dataset for evaluation
            beam_width: Maximum number of candidates to select
            num_threads: Number of threads to use
            min_score: Minimum score when errors occur
            return_scores: Whether to return scores along with parameters
            
        Returns:
            If return_scores is False: List of selected candidates' parameters
            If return_scores is True: Tuple of (list of parameters, list of scores)
        """
        # Store current parameters to restore later
        current_params = {p: copy.deepcopy(p.data) for p in self.optimizer.parameters}
        
        # List to store (score, params) pairs
        scored_candidates = []
        
        # Evaluate each candidate
        for candidate_idx, candidate_params in enumerate(candidates):
            self.optimizer.update(candidate_params)
            
            # Evaluate on validation minibatch using evaluate function
            validation_scores = evaluate(
                self.agent,
                validate_guide,
                validation_mini_dataset['inputs'],
                validation_mini_dataset['infos'],
                min_score=min_score,
                num_threads=num_threads,
                description=f"Validating candidate {candidate_idx+1}/{len(candidates)}"
            )
            
            validation_score = np.mean(validation_scores) if all([s is not None for s in validation_scores]) else -np.inf
            scored_candidates.append((validation_score, candidate_params))
            
            print_color(f"Candidate {candidate_idx+1}: Validation score: {validation_score:.4f}", 'cyan')
        
        # Restore original parameters
        self.optimizer.update(current_params)
        
        # Extract scores for logging
        scores = [score for score, _ in scored_candidates]
        
        # If the number of candidates is less than or equal to beam_width, keep all of them
        if len(scored_candidates) <= beam_width:
            print_color(f"Keeping all {len(scored_candidates)} candidates as num_candidates <= beam_width. Scores: {[f'{s:.4f}' for s in scores]}", 'green')
            selected_params = [params for _, params in scored_candidates]
            if return_scores:
                return selected_params, scores
            return selected_params
        
        # Sort candidates by score (descending)
        sorted_candidates = sorted(scored_candidates, key=lambda x: x[0], reverse=True)
        
        # Select top beam_width candidates
        selected_candidates = sorted_candidates[:beam_width]
        selected_scores = [score for score, _ in selected_candidates]
        selected_params = [params for _, params in selected_candidates]
        
        print_color(f"Selected top {beam_width} beams with scores: {[f'{s:.4f}' for s in selected_scores]}", 'green')
        if return_scores:
            return selected_params, selected_scores
        return selected_params



class BeamsearchHistoryAlgorithm(BeamsearchAlgorithm):
    """
    BeamsearchHistoryAlgorithm enhances BeamsearchAlgorithm by incorporating
    historical parameter-score information into the proposal generation process.

    It maintains a log of previously selected parameter sets and their validation scores.
    This history is then formatted and provided as additional context (feedback)
    during the `expand` phase, aiming to guide the optimizer towards generating
    more informed proposals based on past performance.
    """

    def train(self,
              guide,
              train_dataset,
              *,
              validate_dataset=None,
              validate_guide=None,
              validation_dataset_size=5,
              beam_width=3,
              batch_size=1,
              num_proposals=1,
              max_depth=2,
              num_threads=10,
              max_history_size=10,  # Max number of history entries to keep
              test_frequency=5, # Match the context file value
              # Add other args from parent if needed, or rely on **kwargs
              **kwargs
              ):
        """
        Performs beam search enhanced with parameter history.

        Args:
            max_history_size: Maximum number of (parameter, score) pairs to store
                              in the history log. Defaults to 20.
            top_k: Size of the top-k candidates buffer that persists across depths.
                  Default is 1, which keeps only the best candidate.
            Other args are the same as BeamsearchAlgorithm.train()
        """
        self.total_samples = 0    
        self.min_score = kwargs.get('min_score', 0)
        print_color(f"Running BeamsearchHistoryAlgorithm with beam_width={beam_width}, max_depth={max_depth}, max_history_size={max_history_size}", 'blue')

        # Initialize history log
        self.parameter_history: List[Tuple[Dict, float]] = []
        self.max_history_size = max_history_size

        # Use train dataset for validation if not specified
        validate_dataset = validate_dataset or train_dataset
        validate_guide = validate_guide or guide
        

        # Default validation dataset size
        if validation_dataset_size is None:
            validation_dataset_size = min(10, len(validate_dataset['inputs']))
        print_color(f"Using validation_dataset_size={validation_dataset_size} for intermediate evaluations", 'blue')

        # Store original parameters
        original_params = {p: copy.deepcopy(p.data) for p in self.optimizer.parameters}

        # Dictionary to track metrics
        metrics = {
            'best_validation_scores': [],
            'depth_scores': [],
            'test_scores': [],
            'test_depths': []
        }

        test_dataset = kwargs.get('test_dataset', None)

        # Evaluate initial parameters on test set
        if test_dataset is not None:
            print_color("\n===== Evaluating Initial Parameters =====", 'blue')
            initial_test_scores = evaluate(
                self.agent, guide, test_dataset['inputs'], test_dataset['infos'],
                min_score=self.min_score, num_threads=num_threads,
                description="Evaluating initial parameters on test set"
            )
            initial_test_score = np.mean(initial_test_scores) if all([s is not None for s in initial_test_scores]) else -np.inf
            print_color(f"Initial test score: {initial_test_score:.4f}", 'yellow')
            
            # Log initial test score
            self.logger.log('Initial test score', initial_test_score, 0, color='blue')
            
            metrics['test_scores'].append(initial_test_score)
            metrics['test_depths'].append(1) # Start depth at 1 for consistency

        # Start with a single beam
        beams = [original_params]

        # >>> Main Beam Search Loop <<<
        for depth in range(max_depth):
            print_color(f"\n===== Beam Search Depth {depth+1}/{max_depth} with {len(beams)} beams =====", 'blue')

            # Sample validation minibatch
            validation_xs, validation_infos = self._sample_minibatch(validate_dataset, validation_dataset_size)
            validation_mini_dataset = {'inputs': validation_xs, 'infos': validation_infos}
            print_color(f"Sampled validation minibatch of size {len(validation_xs)} for depth {depth+1}", 'cyan')

            # Expand all current beams
            all_candidates = []
            for beam_idx, beam_params in enumerate(beams):
                print_color(f"Processing beam {beam_idx+1}/{len(beams)}", 'yellow')
                beam_candidates = self.expand( # Calls the overridden expand method
                    beam_params=beam_params, beam_idx=beam_idx, guide=guide,
                    train_dataset=train_dataset, batch_size=batch_size,
                    num_proposals=num_proposals, num_threads=num_threads
                )
                all_candidates.extend(beam_candidates)
                self.total_samples += batch_size
            # Select top candidates
            beams, scores = self.select(
                candidates=all_candidates, validate_guide=validate_guide,
                validation_mini_dataset=validation_mini_dataset, beam_width=beam_width,
                num_threads=num_threads, min_score=self.min_score, return_scores=True
            )
            self.total_samples += validation_dataset_size*len(all_candidates)
            # --- Populate History Log ---
            if scores:
                best_score_this_depth = -np.inf
                for params, score in zip(beams, scores):
                    # params = copy.deepcopy(params)
                    # for name, value in params.items():
                    #     print(f"{name}: {value}")
                    if score > -np.inf: # Only log valid scores
                        # Store deep copies to prevent modification
                        self.parameter_history.append((params, score))
                        best_score_this_depth = max(best_score_this_depth, score)

                # Keep history log bounded
                if len(self.parameter_history) > self.max_history_size:
                    # Keep the ones with most recent
                    self.parameter_history = self.parameter_history[-self.max_history_size:]
                # --- History Log Populated ---

                # Track metrics
                if best_score_this_depth > -np.inf:
                    metrics['best_validation_scores'].append(best_score_this_depth)
                    metrics['depth_scores'].append(scores)
                    print_color(f"Depth {depth+1} - Best validation score: {best_score_this_depth:.4f}", 'green')
                
                    # Log validation metrics
                    step_num = depth + 1
                    self.logger.log('Best validation score', best_score_this_depth, step_num, color='green')
                    self.logger.log('Average validation score', np.mean(scores), step_num, color='cyan')
                    self.logger.log('Min validation score', min(scores), step_num, color='yellow')
                    self.logger.log('Max validation score', max(scores), step_num, color='magenta')
                    self.logger.log('History buffer size', len(self.parameter_history), step_num, color='orange')
                
                    best_idx = scores.index(best_score_this_depth) # Find index of best score
                    best_params = beams[best_idx] # Get corresponding params

                    # Evaluate on test set periodically
                    if test_dataset is not None and ((depth + 1) % test_frequency == 0):
                        self.optimizer.update(best_params) # Use best params from this depth
                        print_color("\nBest parameters at depth {}:".format(depth + 1), 'cyan')

                        for param in self.agent.parameters():
            # Use a try-except block to handle parameter lookup
                            try:
                                # Check if parameter object is directly available as a key
                                if param in best_params:
                                    param_value = best_params[param]
                                # Try to find by name if available
                                elif hasattr(param, 'name') and param.name in best_params:
                                    param_value = best_params[param.name]
                                else:
                                    param_value = "Parameter not found in best_params"
                                
                                # Get the parameter name directly
                                param_name = param.name if hasattr(param, 'name') else str(param)
                                print_color(f"{param_name}: {param_value}", 'blue')
                            except Exception as e:
                                print_color(f"Error accessing parameter {getattr(param, 'name', str(param))}: {e}", 'red')
                                continue
                        test_scores_eval = evaluate(
                            self.agent, guide, test_dataset['inputs'], test_dataset['infos'],
                            min_score=self.min_score, num_threads=num_threads,
                            description=f"Evaluating best parameters at depth {depth+1} on test set"
                        )
                        test_score = np.mean(test_scores_eval) if all([s is not None for s in test_scores_eval]) else -np.inf
                        metrics['test_scores'].append(test_score)
                        metrics['test_depths'].append(depth + 1)
                        print_color(f"Depth {depth+1} - Test score: {test_score:.4f}", 'magenta')
                        
                        # Log test score
                        self.logger.log('Periodic test score', test_score, step_num, color='magenta')

        # >>> End Main Loop <<<

        # Final selection using full validation set
        print_color("\n===== Final Selection Using Full Validation Set =====", 'blue')
        full_validation_dataset = {'inputs': validate_dataset['inputs'], 'infos': validate_dataset['infos']}
        best_beams, final_val_scores = self.select(
            candidates=beams, validate_guide=validate_guide,
            validation_mini_dataset=full_validation_dataset, beam_width=1, # Select only the best
            num_threads=num_threads, min_score=self.min_score, return_scores=True
        )

        final_validation_score = final_val_scores[0] if final_val_scores else -np.inf
        best_params = best_beams[0] if best_beams else original_params # Fallback to original if empty

        # Log final validation score
        final_step = max_depth + 1
        self.logger.log('Final validation score', final_validation_score, final_step, color='blue')
        
        # Apply the best parameters
        self.optimizer.update(best_params)

        # Print final parameters
        print_color("\n===== Final Proposal Candidate Parameters =====", 'magenta')

        # Final evaluation on test set
        final_test_score = None
        if test_dataset is not None:
            final_test_scores_eval = evaluate(
                self.agent, guide, test_dataset['inputs'], test_dataset['infos'],
                min_score=self.min_score, num_threads=num_threads,
                description="Evaluating best beam on test set"
            )
            final_test_score = np.mean(final_test_scores_eval) if all([s is not None for s in final_test_scores_eval]) else -np.inf
            print_color(f"BEST BEAM - Test score: {final_test_score:.4f}", 'green')

            # Log final test score
            self.logger.log('Final test score', final_test_score, final_step, color='green')

        # Save agent if configured
        if kwargs.get('save_frequency', None) is not None and kwargs['save_frequency'] > 0:
             self.save_agent(kwargs.get('save_path', "checkpoints/agent.pkl"), 0)

        # Print test score summary
        if metrics['test_scores']:
            print_color("\n===== Periodic Test Scores Summary =====", 'blue')
            for d, s in zip(metrics['test_depths'], metrics['test_scores']):
                print_color(f"Depth {d}: Test score = {s:.4f}", 'cyan')

        return metrics, final_test_score if final_test_score is not None else -np.inf

    def expand(self,
               beam_params: Dict,
               beam_idx: int,
               guide,
               train_dataset,
               batch_size: int,
               num_proposals: int,
               num_threads: int = None) -> List[Dict]:
        """
        Expands a single candidate into multiple proposals, incorporating history.

        Overrides the parent expand method to augment the feedback provided to the
        optimizer with a summary of historical parameter-score pairs.

        Args: Same as parent expand method.

        Returns: Same as parent expand method.
        """
        # Restore parameters for this beam
        self.optimizer.update(beam_params)

        # Run forward pass on minibatch to get outputs and feedbacks
        xs_batch, infos_batch = self._sample_minibatch(train_dataset, batch_size)

        use_asyncio = self._use_asyncio(num_threads)
        description=f"Forward pass (beam {beam_idx+1}, batch size: {len(xs_batch)})"
        if use_asyncio:
            outputs = async_run([self.forward]*len(xs_batch),
                               [(self.agent, x, guide, info) for x, info in zip(xs_batch, infos_batch)],
                               max_workers=num_threads, description=description)
        else:
            outputs = [self.forward(self.agent, x, guide, info) for x, info in zip(xs_batch, infos_batch)]

        # Prepare original feedback
        scores, targets, feedbacks = [], [], []
        for target, score, feedback_item in outputs:
            scores.append(score)
            targets.append(target)
            feedbacks.append(feedback_item)
        target = batchify(*targets)
        original_feedback = batchify(*feedbacks).data # Assuming .data gives the relevant part

        # --- History Injection ---
        history_prompt = "\n--- History Context ---\n"
        history_prompt += "Consider the following previously selected parameter sets and their validation scores when generating proposals:\n"
        if not self.parameter_history:
            history_prompt += "(No history available yet)\n"
        else:
            # Format history (e.g., last N entries)
            # Sorting by score might be useful: sorted_history = sorted(self.parameter_history, key=lambda item: item[1], reverse=True)
            display_history = self.parameter_history # Or sorted_history[:self.max_history_size]
            for i, (hist_params, hist_score) in enumerate(display_history):
                 # Format parameters nicely
                param_parts = []
                for k, v in hist_params.items():
                    key_name = getattr(k, 'name', str(k)) # Get name attr if Parameter object
                    if isinstance(v, (float, np.floating)):
                         param_parts.append(f"{key_name}: {v:.4f}")
                    elif isinstance(v, (np.ndarray, list)) and len(v) > 5: # Truncate long lists/arrays
                         param_parts.append(f"{key_name}: [{', '.join(map(str, v[:2]))}...{str(v[-1])}]")
                    else:
                         param_parts.append(f"{key_name}: {v}")
                param_str = ", ".join(param_parts)
                history_prompt += f"  Attempt {i+1} (Score: {hist_score:.4f}): {{{param_str}}}\n"

        # Combine history with original feedback
        # This assumes the optimizer can handle string feedback or a dict.
        # Adjust based on how your specific optimizer/trace uses feedback.
        augmented_feedback: Union[str, Dict]
        if isinstance(original_feedback, str):
             augmented_feedback = f"--- Current Feedback ---\n{original_feedback}\n{history_prompt}"
        elif isinstance(original_feedback, dict):
            # Add history as a separate key, preserving original structure
             augmented_feedback = original_feedback.copy()
             augmented_feedback['history_context'] = history_prompt
             # Ensure original feedback text (if any) is still prominent
             if 'feedback' in augmented_feedback:
                 augmented_feedback['feedback'] = f"{augmented_feedback['feedback']}\n{history_prompt}"
             elif 'prompt' in augmented_feedback: # Adapt if feedback is under 'prompt' key
                 augmented_feedback['prompt'] = f"{augmented_feedback['prompt']}\n{history_prompt}"
             else: # Fallback if structure unknown
                 augmented_feedback['raw_feedback'] = original_feedback

        else:
            # Attempt to stringify other types, may need refinement
            try:
                augmented_feedback = f"--- Current Feedback ---\n{str(original_feedback)}\n{history_prompt}"
                print_color(f"Warning: Combined non-string/dict feedback with history prompt.", "yellow")
            except Exception as e:
                print_color(f"Error combining feedback with history: {e}. Using original feedback.", "red")
                augmented_feedback = original_feedback # Fallback

        # --- End History Injection ---


        # Backward pass using the augmented feedback
        self.optimizer.zero_feedback()
        self.optimizer.backward(target, augmented_feedback) # Pass augmented feedback here

        # Generate multiple proposals using optimizer.step
        step_kwargs = dict(bypassing=True, verbose='output')
        candidates = []
        description_step=f"Generating {num_proposals} proposals for beam {beam_idx+1} (with history)"
        if use_asyncio:
            update_dicts = async_run([self.optimizer.step]*num_proposals,
                                    kwargs_list=[step_kwargs] * num_proposals,
                                    max_workers=num_threads,
                                    description=description_step)
        else:
            update_dicts = [self.optimizer.step(**step_kwargs) for _ in range(num_proposals)]

        # Collect all valid proposals
        for update_dict in update_dicts:
            if len(update_dict) > 0:
                # Make sure update_dict contains all parameters from beam_params
                # Add any missing parameters from beam_params to update_dict
                for param_key, param_value in beam_params.items():
                    if param_key not in update_dict:
                        update_dict[param_key] = param_value
                candidates.append(update_dict)

        # Also include the original beam parameters as a candidate
        candidates.append(beam_params)

        return candidates

