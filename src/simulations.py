"""
Simulation module for generating synthetic survey responses using Large Language Models (LLMs).

This module provides functionality to:
1. Generate synthetic answers to survey questions using various LLM APIs (OpenAI, Anthropic, TogetherAI)
2. Process and clean the raw responses to extract structured answers
3. Handle retries, error recovery, and concurrent API calls
4. Support two datasets: EEDI (educational assessment) and OpinionQA (opinion polling)

The module uses asynchronous programming for efficient concurrent API calls and includes
robust error handling with exponential backoff retry logic.
"""

import re
import os
import json
import time
import random
import numpy as np
from difflib import SequenceMatcher
from tqdm import tqdm
import asyncio
import aiohttp


# Path configuration
# ROOT_PATH: Path to the project root directory (one level up from this file)
ROOT_PATH = os.path.join('..')
# DATA_PATH: Path to the data directory containing datasets
DATA_PATH = os.path.join(ROOT_PATH, 'data')

# Mapping from internal LLM file names to actual API model names
# Keys: Internal identifiers used in file names and function calls
# Values: Actual model names/identifiers used by the respective API platforms
LLM_FILE_NAME_TO_MODEL_NAME = {
    'claude-3.5-haiku': 'claude-3-5-haiku-20241022',  # Anthropic API
    'deepseek-v3': 'deepseek-ai/DeepSeek-V3',  # TogetherAI API
    'gpt-3.5-turbo': 'gpt-3.5-turbo',  # OpenAI API
    'gpt-4o-mini': 'gpt-4o-mini',  # OpenAI API
    'gpt-4o': 'gpt-4o',  # OpenAI API
    'gpt-5-mini': 'gpt-5-mini',  # OpenAI API
    'llama-3.3-70B-instruct-turbo': 'meta-llama/Llama-3.3-70B-Instruct-Turbo',  # TogetherAI API
    'mistral-7B-instruct-v0.3': 'mistralai/Mistral-7B-Instruct-v0.3',  # TogetherAI API
}

async def generate_responses(
    api_platform: str,
    api_key: str,
    llm: str,
    dataset_name: str,
    first_synthetic_profile_id: int,
    num_of_synthetic_answers: int,
    folder_name: str,
    max_concurrent_requests: int = 10,
    max_retries: int = 3
) -> None:
    """
    Generate synthetic survey responses using LLM APIs with asynchronous processing.
    
    This function orchestrates the entire pipeline:
    1. Loads the dataset and existing results (if any)
    2. Generates synthetic answers by calling LLM APIs concurrently
    3. Saves raw responses incrementally (after each question)
    4. Post-processes responses to extract structured answers
    5. Saves cleaned results to JSON files
    6. Generates and saves a random baseline for comparison
    
    The function supports resuming from existing results - if raw results already exist,
    it will only process questions that haven't been completed yet.
    
    Args:
        api_platform (str): API platform to use. Must be one of:
            - 'openai': OpenAI API (GPT models)
            - 'anthropic': Anthropic API (Claude models)
            - 'togetherai': TogetherAI API (various open-source models)
        api_key (str): API key for authentication with the chosen platform
        llm (str): LLM model identifier. Must be a key in LLM_FILE_NAME_TO_MODEL_NAME.
            Examples: 'gpt-4o', 'claude-3.5-haiku', 'mistral-7B-instruct-v0.3'
        dataset_name (str): Dataset name. Must be one of:
            - 'EEDI': Educational assessment dataset
            - 'OpinionQA': Opinion polling dataset
        first_synthetic_profile_id (int): The id of the first synthetic profile to use for the question.
            Should be an integer between 0 and 200 for both datasets.
        num_of_synthetic_answers (int): Number of synthetic answers to generate per question.
            first_synthetic_profile_id + num_of_synthetic_answers should not exceed the number of available synthetic profiles (200 for both datasets) for each question.
        folder_name (str): Name of the folder to save results in (e.g., 'synthetic_answers').
            Results will be saved in: data/{dataset_name}/{folder_name}/raw/ and
            data/{dataset_name}/{folder_name}/clean/
            A random baseline will also be saved to: data/{dataset_name}/{folder_name}/clean/random.json
        max_concurrent_requests (int, optional): Maximum number of concurrent API requests.
            Defaults to 10. Higher values increase throughput but may hit rate limits.
        max_retries (int, optional): Maximum number of retry attempts for failed API calls.
            Defaults to 3. Uses exponential backoff between retries (1s, 2s, 4s, ...).
    
    Returns:
        None: Results are saved to disk. The function prints progress messages and
        a summary of the total time taken.
    
    Raises:
        AssertionError: If api_platform, dataset_name, or llm are not valid values.
    
    Example:
        ```python
        import asyncio
        await generate_responses(
            api_platform='openai',
            api_key='sk-...',
            llm='gpt-4o',
            dataset_name='OpinionQA',
            first_synthetic_profile_id=0,
            num_of_synthetic_answers=100,
            folder_name='synthetic_answers',
            max_concurrent_requests=10,
            max_retries=3
        )
        ```
    
    Note:
        - Raw results are saved incrementally after each question completes
        - If the function is interrupted, it can be resumed and will skip already-completed questions
        - Error responses are saved as strings starting with "ERROR:"
        - Post-processing extracts structured answers from raw responses using regex and similarity matching
        - A random baseline is generated with 2 * num_of_synthetic_answers responses per question:
          * EEDI: random binary correctness scores from {0, 1}
          * OpinionQA: random opinion scores from {-1, -1/3, 0, 1/3, 1}
    """

    # Record start time for performance tracking
    start_time = time.time()

    # Validate input parameters
    assert api_platform in {'openai', 'anthropic', 'togetherai'}, \
        f"Invalid api_platform: {api_platform}. Must be 'openai', 'anthropic', or 'togetherai'."
    assert dataset_name in {'EEDI', 'OpinionQA'}, \
        f"Invalid dataset_name: {dataset_name}. Must be 'EEDI' or 'OpinionQA'."
    assert llm in LLM_FILE_NAME_TO_MODEL_NAME.keys(), \
        f"Invalid llm: {llm}. Must be one of {list(LLM_FILE_NAME_TO_MODEL_NAME.keys())}."
    
    # Get the actual model name for the API (may differ from internal identifier)
    model_name = LLM_FILE_NAME_TO_MODEL_NAME[llm]

    print(f'Generating {num_of_synthetic_answers} responses for questions in {dataset_name} with {llm}, starting from synthetic profile {first_synthetic_profile_id}...')

    # ========================================================================
    # SET UP FILE PATHS
    # ========================================================================
    # Construct paths based on dataset name
    if dataset_name == 'EEDI':
        specific_data_path = os.path.join(DATA_PATH, dataset_name)
        dataset_path = os.path.join(specific_data_path, 'eedi_data.json')
    elif dataset_name == 'OpinionQA':
        specific_data_path = os.path.join(DATA_PATH, dataset_name)
        dataset_path = os.path.join(specific_data_path, 'opinionqa_data.json')
    
    # Set up output directory structure:
    # data/{dataset_name}/{folder_name}/raw/   - for raw LLM responses
    # data/{dataset_name}/{folder_name}/clean/ - for processed/cleaned responses
    folder_for_raw_and_clean_results_path = os.path.join(specific_data_path, folder_name)
    raw_path = os.path.join(folder_for_raw_and_clean_results_path, 'raw')
    clean_path = os.path.join(folder_for_raw_and_clean_results_path, 'clean')
    
    # Create directories if they don't exist
    if not os.path.exists(raw_path):
        os.makedirs(raw_path)
    if not os.path.exists(clean_path):
        os.makedirs(clean_path)
    
    # Full paths to output files
    raw_file_path = os.path.join(raw_path, f'{llm}.json')
    clean_file_path = os.path.join(clean_path, f'{llm}.json')
    random_file_path = os.path.join(clean_path, 'random.json')

    print(f'Raw results will be saved to {raw_file_path}')
    print(f'Clean results will be saved to {clean_file_path}')
    print(f'Random baseline will be saved to {random_file_path}')

    # ========================================================================
    # LOAD DATASET
    # ========================================================================
    # Load the dataset JSON file containing questions, synthetic profiles, etc.
    # See README.md for the dataset structure.
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    # ========================================================================
    # SET UP INSTRUCTION PROMPT
    # ========================================================================
    # System instruction that will be prepended to each query to guide the LLM
    instruction = "You are simulating the behaviors of humans with certain specified characteristics to help with a survey study."

    # ========================================================================
    # LOAD EXISTING RESULTS (FOR RESUME FUNCTIONALITY)
    # ========================================================================
    # Check if raw results already exist - allows resuming interrupted runs
    # Structure: {question_key (str): [list of response strings]}
    synthetic_answers_raw = {}  # key: question id (str), value: list of synthetic answers (list of strings)
    if os.path.exists(raw_file_path):
        with open(raw_file_path, 'r') as f:
            synthetic_answers_raw = json.load(f)
        print(f'Found and loaded existing results at {raw_file_path}.')
    else:
        print(f'No existing results found at {raw_file_path}. Starting from scratch.')

    # ========================================================================
    # SET UP CONCURRENCY CONTROL
    # ========================================================================
    # Semaphore limits the number of concurrent API requests to avoid hitting rate limits
    # This ensures we don't exceed max_concurrent_requests simultaneous API calls
    semaphore = asyncio.Semaphore(max_concurrent_requests)

    async def make_api_call(query: str, session: aiohttp.ClientSession) -> str:
        """
        Make a single API call to an LLM with retry logic and exponential backoff.
        
        This function handles API calls to different LLM providers (OpenAI, Anthropic, TogetherAI)
        with automatic retry on failures. It uses a semaphore to limit concurrent requests
        and implements exponential backoff between retry attempts.
        
        Args:
            query (str): The prompt/query to send to the LLM. This is typically a synthetic
                profile prompt that asks the LLM to simulate a human response.
            session (aiohttp.ClientSession): The aiohttp session for making HTTP requests.
                This is reused across multiple calls for efficiency.
        
        Returns:
            str: The LLM's response text. If all retries fail, returns an error string
                starting with "ERROR: Failed after {max_retries} attempts - {error_message}".
        
        Note:
            - Uses exponential backoff: waits 2^attempt seconds between retries (1s, 2s, 4s, ...)
            - All exceptions are caught and retried (network errors, timeouts, API errors, etc.)
            - The function respects the semaphore limit set by max_concurrent_requests
            - Different API platforms have different request formats and response structures
            - Can be modified to include other LLM APIs or API-specific features (temperature, max_tokens, etc.)
        """
        # Acquire semaphore to limit concurrent requests (released automatically when done)
        async with semaphore:
            last_exception = None
            
            # Retry loop: attempt up to max_retries times
            for attempt in range(max_retries):
                try:
                    # ============================================================
                    # OPENAI API CALL
                    # ============================================================
                    if api_platform == 'openai':
                        # OpenAI uses a system/user message format
                        payload = {
                            "model": model_name,  # e.g., "gpt-4o", "gpt-3.5-turbo"
                            "messages": [
                                {"role": "system", "content": instruction},  # System instruction
                                {"role": "user", "content": query}  # User query (synthetic profile prompt)
                            ],
                        }
                        headers = {
                            "Authorization": f"Bearer {api_key}",  # Bearer token authentication
                            "Content-Type": "application/json"
                        }
                        
                        # Make POST request to OpenAI API endpoint
                        async with session.post(
                            "https://api.openai.com/v1/chat/completions",
                            headers=headers,
                            json=payload,
                            timeout=aiohttp.ClientTimeout(total=60)  # 60 second timeout
                        ) as response:
                            # Check HTTP status code before parsing
                            if response.status != 200:
                                error_text = await response.text()
                                raise Exception(f"OpenAI API returned status {response.status}: {error_text}")
                            
                            # Parse JSON response
                            result = await response.json()
                            
                            # Check for API error response (rate limits, server errors, etc.)
                            if 'error' in result:
                                error_info = result['error']
                                error_msg = error_info.get('message', str(error_info))
                                error_type = error_info.get('type', 'unknown_error')
                                raise Exception(f"OpenAI API error ({error_type}): {error_msg}")
                            
                            # Validate response structure before accessing nested keys
                            if 'choices' not in result:
                                raise Exception(f"OpenAI API response missing 'choices' key. Response: {result}")
                            if not result['choices'] or len(result['choices']) == 0:
                                raise Exception(f"OpenAI API response has empty 'choices' array. Response: {result}")
                            if 'message' not in result['choices'][0]:
                                raise Exception(f"OpenAI API response missing 'message' in choices[0]. Response: {result}")
                            if 'content' not in result['choices'][0]['message']:
                                raise Exception(f"OpenAI API response missing 'content' in message. Response: {result}")
                            
                            # Extract the text content from the response
                            # OpenAI returns: {"choices": [{"message": {"content": "..."}}]}
                            return result['choices'][0]['message']['content']

                    # ============================================================
                    # ANTHROPIC API CALL
                    # ============================================================
                    elif api_platform == 'anthropic':
                        # Anthropic doesn't use separate system messages - combine instruction with query
                        payload = {
                            "model": model_name,  # e.g., "claude-3-5-haiku-20241022"
                            "messages": [
                                {"role": "user", "content": instruction + "\n\n" + query}
                            ],
                            "max_tokens": 4096  # Required by Anthropic API
                        }
                        headers = {
                            "x-api-key": api_key,  # Anthropic uses x-api-key header
                            "anthropic-version": "2023-06-01",  # Required API version header
                            "Content-Type": "application/json"
                        }
                        
                        # Make POST request to Anthropic API endpoint
                        async with session.post(
                            "https://api.anthropic.com/v1/messages",
                            headers=headers,
                            json=payload,
                            timeout=aiohttp.ClientTimeout(total=60)  # 60 second timeout
                        ) as response:
                            # Check HTTP status code before parsing
                            if response.status != 200:
                                error_text = await response.text()
                                raise Exception(f"Anthropic API returned status {response.status}: {error_text}")
                            
                            # Parse JSON response
                            result = await response.json()
                            
                            # Check for API error response
                            if 'error' in result:
                                error_info = result['error']
                                error_msg = error_info.get('message', str(error_info))
                                error_type = error_info.get('type', 'unknown_error')
                                raise Exception(f"Anthropic API error ({error_type}): {error_msg}")
                            
                            # Validate response structure
                            if 'content' not in result:
                                raise Exception(f"Anthropic API response missing 'content' key. Response: {result}")
                            if not result['content'] or len(result['content']) == 0:
                                raise Exception(f"Anthropic API response has empty 'content' array. Response: {result}")
                            if 'text' not in result['content'][0]:
                                raise Exception(f"Anthropic API response missing 'text' in content[0]. Response: {result}")
                            
                            # Extract the text content from the response
                            # Anthropic returns: {"content": [{"text": "..."}]}
                            return result['content'][0]['text']

                    # ============================================================
                    # TOGETHERAI API CALL
                    # ============================================================
                    elif api_platform == 'togetherai':
                        # TogetherAI uses OpenAI-compatible format
                        payload = {
                            "model": model_name,  # e.g., "meta-llama/Llama-3.3-70B-Instruct-Turbo"
                            "messages": [
                                {"role": "system", "content": instruction},  # System instruction
                                {"role": "user", "content": query}  # User query
                            ],
                        }
                        headers = {
                            "Authorization": f"Bearer {api_key}",  # Bearer token authentication
                            "Content-Type": "application/json"
                        }
                        
                        # Make POST request to TogetherAI API endpoint
                        async with session.post(
                            "https://api.together.xyz/v1/chat/completions",
                            headers=headers,
                            json=payload,
                            timeout=aiohttp.ClientTimeout(total=60)  # 60 second timeout
                        ) as response:
                            # Check HTTP status code before parsing
                            if response.status != 200:
                                error_text = await response.text()
                                raise Exception(f"TogetherAI API returned status {response.status}: {error_text}")
                            
                            # Parse JSON response
                            result = await response.json()
                            
                            # Check for API error response
                            if 'error' in result:
                                error_info = result['error']
                                error_msg = error_info.get('message', str(error_info))
                                error_type = error_info.get('type', 'unknown_error')
                                raise Exception(f"TogetherAI API error ({error_type}): {error_msg}")
                            
                            # Validate response structure
                            if 'choices' not in result:
                                raise Exception(f"TogetherAI API response missing 'choices' key. Response: {result}")
                            if not result['choices'] or len(result['choices']) == 0:
                                raise Exception(f"TogetherAI API response has empty 'choices' array. Response: {result}")
                            if 'message' not in result['choices'][0]:
                                raise Exception(f"TogetherAI API response missing 'message' in choices[0]. Response: {result}")
                            if 'content' not in result['choices'][0]['message']:
                                raise Exception(f"TogetherAI API response missing 'content' in message. Response: {result}")
                            
                            # Extract the text content from the response
                            # TogetherAI returns: {"choices": [{"message": {"content": "..."}}]}
                            return result['choices'][0]['message']['content']

                # ============================================================
                # ERROR HANDLING AND RETRY LOGIC
                # ============================================================
                except Exception as e:
                    # Store the exception for potential use in final error message
                    last_exception = e
                    
                    # If we haven't exhausted all retries, wait and try again
                    if attempt < max_retries - 1:
                        # Exponential backoff: wait 2^attempt seconds
                        # Attempt 0: wait 1s, Attempt 1: wait 2s, Attempt 2: wait 4s, etc.
                        await asyncio.sleep(2 ** attempt)
                        continue  # Try again
                    else:
                        # All retries exhausted - return error message
                        return f"ERROR: Failed after {max_retries} attempts - {str(e)}"
            
            # This should never be reached (all paths should return or continue)
            # But included as a safety net
            return f"ERROR: Failed after {max_retries} attempts - {str(last_exception)}"

    async def process_question(question_key: str, session: aiohttp.ClientSession) -> None:
        """
        Process all synthetic answers for a single question concurrently.
        
        This function:
        1. Retrieves synthetic profiles for the question
        2. Creates concurrent API call tasks for each profile
        3. Executes all API calls concurrently using asyncio.gather
        4. Collects and saves results incrementally (after each question)
        
        Args:
            question_key (str): The key identifying the question in the dataset.
                Used to look up question data and synthetic profiles.
            session (aiohttp.ClientSession): The aiohttp session for making HTTP requests.
        
        Returns:
            None: Results are saved to synthetic_answers_raw dictionary and written to disk.
        
        Note:
            - All API calls for a question are executed concurrently for efficiency
            - Results are saved immediately after processing each question (incremental saves)
            - Error responses (starting with "ERROR:") are included in results and logged
            - If fewer profiles are available than requested, uses all available profiles
        """
        answers_for_question = []  # List to store all responses for this question

        # ========================================================================
        # PREPARE SYNTHETIC PROFILES AND VALIDATE AVAILABILITY
        # ========================================================================
        # Get the list of synthetic profiles for this question
        # Each profile is a dict with a 'PROMPT' key containing the prompt text
        synthetic_profiles = dataset[question_key]['synthetic_profile']
        num_profiles = len(synthetic_profiles)
        
        # Validate that we have enough profiles
        # If requested more than available, use all available and warn user
        if num_profiles == 0:
            raise ValueError(f'No synthetic profiles available for Question {question_key}.')
        if first_synthetic_profile_id < 0 or first_synthetic_profile_id >= num_profiles:
            raise ValueError(f'Invalid first_synthetic_profile_id: {first_synthetic_profile_id}. Should be an integer between 0 and {num_profiles - 1} (inclusive) for Question {question_key}.')
        if first_synthetic_profile_id + num_of_synthetic_answers > num_profiles:
            print(f'Warning: Requested {num_of_synthetic_answers} answers starting from profile {first_synthetic_profile_id} but only {num_profiles - first_synthetic_profile_id} profiles available for Question {question_key}. Obtaining {num_profiles - first_synthetic_profile_id} synthetic answers.')
            num_to_use = num_profiles - first_synthetic_profile_id
        else:
            num_to_use = num_of_synthetic_answers
        
        # ========================================================================
        # CREATE CONCURRENT API CALL TASKS
        # ========================================================================
        # Create a list of async tasks, one for each synthetic profile
        tasks = []
        for j in range(num_to_use):
            # Extract the prompt from the synthetic profile
            # synthetic_profile is a list of dicts, each with a 'PROMPT' key
            query = synthetic_profiles[first_synthetic_profile_id + j]['PROMPT']
            # Create an async task for this API call
            task = make_api_call(query, session)
            tasks.append(task)

        # ========================================================================
        # EXECUTE ALL API CALLS CONCURRENTLY
        # ========================================================================
        # Use asyncio.gather to execute all tasks concurrently
        # This waits for all API calls to complete and returns a list of results
        results = await asyncio.gather(*tasks)

        # ========================================================================
        # CHECK FOR ERRORS
        # ========================================================================
        # Identify any results that are error messages (start with "ERROR:")
        # Safely check if result is a string before calling startswith
        error_indices = [i for i, result in enumerate(results) if isinstance(result, str) and result.startswith('ERROR:')]
        if len(error_indices) > 0:
            print(f'Warning: {len(error_indices)} errors occurred for Question {question_key}.')
            # Print details of each error
            for i in error_indices:
                print(f'Error at index {i}: {results[i]}')
        
        # ========================================================================
        # COLLECT RESULTS
        # ========================================================================
        # Extend the answers list with all results (both successful and error responses)
        # results is a list of strings (LLM responses or error messages)
        answers_for_question.extend(results)
        
        # Store results in the global dictionary
        synthetic_answers_raw[question_key] = answers_for_question

        # ========================================================================
        # SAVE RESULTS INCREMENTALLY
        # ========================================================================
        # Save results after completing each question (allows resuming if interrupted)
        # This ensures progress is not lost if the process is interrupted
        with open(raw_file_path, 'w') as f:
            json.dump(synthetic_answers_raw, f)

    # ========================================================================
    # PROCESS ALL QUESTIONS
    # ========================================================================
    # Calculate how many questions still need to be processed
    # (questions that aren't already in synthetic_answers_raw)
    num_of_questions_to_process = len(dataset.keys()) - len(synthetic_answers_raw.keys())
    
    if num_of_questions_to_process == 0:
        print('All questions have been processed.')
    else:
        print(f'Start processing {num_of_questions_to_process} questions...')
        # Create an aiohttp session for making API requests
        # The session is reused across all questions for efficiency
        async with aiohttp.ClientSession() as session:
            # Create tasks for all questions that need processing
            tasks = []
            for question_key in dataset.keys():
                # Skip questions that have already been processed
                if question_key not in synthetic_answers_raw:
                    task = process_question(question_key, session)
                    tasks.append(task)

            # Execute all questions sequentially (one question at a time)
            # Note: Questions are processed sequentially, but within each question,
            # all API calls for that question are concurrent
            # tqdm provides a progress bar to track progress
            for task in tqdm(tasks, desc='Processing questions', total=len(tasks)):
                await task

    print(f'Simulation completed. Results saved to {raw_file_path}.')
    print(f'Post-processing raw responses into clean responses compatible with {dataset_name}...')
    
    # Check if clean results already exist (will be overwritten)
    if os.path.exists(clean_file_path):
        print(f'Clean results already exist at {clean_file_path}. Proceeding to overwrite.')
    else:
        print(f'Clean results do not exist. Starting from scratch.')

    # ========================================================================
    # AUXILIARY FUNCTION: SIMILARITY CALCULATION
    # ========================================================================
    def similarity(a: str, b: str) -> float:
        """
        Calculate the similarity ratio between two strings using SequenceMatcher.
        
        This function is used to match LLM responses to expected answer choices when
        the response doesn't contain the expected format (e.g., no double brackets).
        
        Args:
            a (str): First string to compare
            b (str): Second string to compare
        
        Returns:
            float: Similarity ratio between 0.0 (no similarity) and 1.0 (identical).
                Higher values indicate greater similarity.
        
        Example:
            similarity("Option A", "Option A")  # Returns 1.0
            similarity("Option A", "Option B")  # Returns ~0.67 (partial match)
        """
        return SequenceMatcher(None, a, b).ratio()

    def process_responses_EEDI(synthetic_answers_raw: dict) -> None:
        """
        Process and clean EEDI dataset synthetic responses.
        
        This function:
        1. Extracts answer letters (A, B, C, D) from raw LLM responses
        2. Converts letters to numbers (A=1, B=2, C=3, D=4)
        3. Calculates correctness by comparing to ground truth answers
        4. Saves cleaned results (binary correctness scores) to JSON file
        
        Args:
            synthetic_answers_raw (dict): Dictionary mapping question keys to lists of
                raw LLM response strings. Structure: {question_key: [response1, response2, ...]}
        
        Returns:
            None: Results are saved to clean_file_path.
        
        Processing Steps:
            1. Extract answer from double brackets: [[A]], [[B]], etc.
            2. If no brackets, extract capital letters from response
            3. If no letters found, use similarity matching against answer_to_letter mapping
            4. Convert letters to numbers (A=1, B=2, C=3, D=4)
            5. Compare to ground truth to determine correctness (1=correct, 0=incorrect)
        
        Output Format:
            {question_key: [1, 0, 1, 1, 0, ...]}  # List of binary correctness scores
        """
        # Dictionary to store cleaned results (correctness scores)
        synthetic_answers_clean = {}

        # Dictionary to store extracted answer letters before conversion to numbers
        synthetic_answers = {}

        # ========================================================================
        # STEP 1: EXTRACT ANSWER LETTERS FROM RAW RESPONSES
        # ========================================================================
        for question_key in tqdm(synthetic_answers_raw.keys(), desc='Extracting answer letters'):
            synthetic_answers[question_key] = []

            for response in synthetic_answers_raw[question_key]:
                # If start with 'ERROR:', return random answer letter
                if response.startswith('ERROR:'):
                    synthetic_answers[question_key].append(random.choice(['A', 'B', 'C', 'D']))
                    continue
                
                # ============================================================
                # EXTRACT TEXT FROM DOUBLE BRACKETS
                # ============================================================
                # Look for patterns like [[A]], [[B]], [[Answer: C]], etc.
                # The regex r'\[\[(.*?)\]' captures everything between double brackets
                answer = re.findall(r'\[\[(.*?)\]', response)

                if len(answer) == 0:
                    # No double brackets found - use full response for similarity matching
                    answer_text = response
                    answer_letter = []
                else:
                    # Extract the text inside the double brackets
                    answer_text = answer[0]
                    # Extract any capital letters from the bracketed text (A, B, C, D)
                    answer_letter = re.findall(r'[A-Z]', answer_text)

                # ============================================================
                # DETERMINE ANSWER LETTER
                # ============================================================
                if len(answer_letter) == 0:
                    # No capital letter found - use similarity matching
                    # Get the mapping of answer text to letters for this question
                    ans_to_letter_i = dataset[question_key]['answer_to_letter']
                    # Calculate similarity scores against all possible answers
                    sims = [similarity(answer_text, ans_to_letter_i[key]) for key in ans_to_letter_i.keys()]
                    # Find the answer with the highest similarity
                    max_sim_ind = np.argmax(sims)
                    # Get the corresponding letter (A, B, C, or D)
                    answer_letter = list(ans_to_letter_i.values())[max_sim_ind]
                else:
                    # Use the first capital letter found
                    answer_letter = answer_letter[0]
                
                synthetic_answers[question_key].append(answer_letter)

        # ========================================================================
        # STEP 2: CONVERT LETTERS TO NUMBERS
        # ========================================================================
        # EEDI dataset uses numbers 1, 2, 3, 4 to represent A, B, C, D
        dict_letter_to_number = {'A': 1, 'B': 2, 'C': 3, 'D': 4}

        synthetic_answers_number = {}
        for question_key in tqdm(synthetic_answers_raw.keys(), desc='Converting letters to numbers'):
            answers_num_i = []
            for answer_letter in synthetic_answers[question_key]:
                # Handle invalid letters (shouldn't happen, but defensive programming)
                if answer_letter not in dict_letter_to_number:
                    # Use similarity matching to find the closest valid letter
                    admissible_letters = list(dict_letter_to_number.keys())
                    sims = [similarity(answer_letter, key) for key in admissible_letters]
                    max_sim_ind = np.argmax(sims)
                    answer_letter = admissible_letters[max_sim_ind]
                # Convert letter to number
                answers_num_i.append(dict_letter_to_number[answer_letter])
            synthetic_answers_number[question_key] = answers_num_i

        # ========================================================================
        # STEP 3: CALCULATE CORRECTNESS
        # ========================================================================
        # Compare each answer to the ground truth answer for the question
        for question_key in tqdm(synthetic_answers_raw.keys(), desc='Calculating correctness'):
            iscorrect_i = []
            # Get the correct answer for this question from the dataset
            correct_answer = dataset[question_key]['answer']
            for answer_number in synthetic_answers_number[question_key]:
                # 1 if correct, 0 if incorrect
                iscorrect_i.append(int(answer_number == correct_answer))
            synthetic_answers_clean[question_key] = iscorrect_i

        # ========================================================================
        # STEP 4: SAVE CLEANED RESULTS
        # ========================================================================
        # Save the cleaned results (binary correctness scores) to JSON file
        with open(clean_file_path, 'w') as f:
            json.dump(synthetic_answers_clean, f)

    def process_responses_OpinionQA(synthetic_answers_raw: dict) -> None:
        """
        Process and clean OpinionQA dataset synthetic responses.
        
        This function:
        1. Extracts answer numbers (1-5) from raw LLM responses
        2. Converts numbers to numeric scores using the mapping:
           1 -> 1.0, 2 -> 1/3, 3 -> -1/3, 4 -> -1.0, 5 -> 0.0
        3. If no number is found, uses similarity matching against choices_to_numeric mapping
        4. Saves cleaned results (numeric scores) to JSON file
        
        Args:
            synthetic_answers_raw (dict): Dictionary mapping question keys to lists of
                raw LLM response strings. Structure: {question_key: [response1, response2, ...]}
        
        Returns:
            None: Results are saved to clean_file_path.
        
        Processing Steps:
            1. Extract answer from double brackets: [[1]], [[2]], etc.
            2. If no brackets, extract numbers from response text
            3. If no number found or invalid number, use similarity matching against choices_to_numeric
            4. Map numbers to numeric scores: 1=1.0, 2=1/3, 3=-1/3, 4=-1.0, 5=0.0
        
        Output Format:
            {question_key: [1.0, 0.333..., -0.333..., -1.0, 0.0, ...]}  # List of numeric scores
        
        Note:
            The numeric scores represent opinion strength on a scale from -1 (strongly negative)
            to 1 (strongly positive), with 0 representing neutral or refused responses.
        """
        # Dictionary to store cleaned results (numeric scores)
        synthetic_answers_clean = {}

        for question_key in tqdm(synthetic_answers_raw.keys(), desc='Converting numbers to numeric scores'):
            numeric_i = []  # List to store numeric scores for this question
            
            for response in synthetic_answers_raw[question_key]:
                # If start with 'ERROR:', return random answer number
                if response.startswith('ERROR:'):
                    numeric_i.append(random.choice([-1, -1/3, 0, 1/3, 1]))
                    continue
                
                # ============================================================
                # EXTRACT TEXT FROM DOUBLE BRACKETS
                # ============================================================
                # Look for patterns like [[1]], [[2]], [[Answer: 3]], etc.
                # The regex r'\[\[(.*?)\]' captures everything between double brackets
                answer = re.findall(r'\[\[(.*?)\]', response)

                if len(answer) == 0:
                    # No double brackets found - use full response for similarity matching
                    answer_text = response
                    answer_num = []
                else:
                    # Extract the text inside the double brackets
                    answer_text = answer[0]
                    # Extract any numbers from the bracketed text (1, 2, 3, 4, 5)
                    answer_num = re.findall(r'\d+', answer_text)

                # ============================================================
                # DETERMINE NUMERIC SCORE
                # ============================================================
                if len(answer_num) == 0:
                    # No number found - use similarity matching
                    # Get the mapping of choice text to numeric scores for this question
                    ans_to_num_i = dataset[question_key]['choices_to_numeric']
                    # Calculate similarity scores against all possible choices
                    sims = [similarity(answer_text, key) for key in ans_to_num_i.keys()]
                    # Find the choice with the highest similarity
                    max_sim_ind = np.argmax(sims)
                    # Get the corresponding numeric score
                    answer_num = list(ans_to_num_i.values())[max_sim_ind]
                else:
                    # Number found - map to numeric score
                    # Mapping: 1 -> 1.0, 2 -> 1/3, 3 -> -1/3, 4 -> -1.0, 5 -> 0.0
                    num_to_num = {'1': 1, '2': 1/3, '3': -1/3, '4': -1, '5': 0}
                    answer_num_str = str(answer_num[0])
                    
                    if answer_num_str not in num_to_num:
                        # Invalid number (not 1-5) - use similarity matching as fallback
                        admissible_numbers = list(num_to_num.keys())
                        sims = [similarity(answer_num_str, key) for key in admissible_numbers]
                        max_sim_ind = np.argmax(sims)
                        answer_num = num_to_num[admissible_numbers[max_sim_ind]]
                    else:
                        # Valid number - convert to numeric score
                        answer_num = num_to_num[answer_num_str]

                # Add the numeric score to the list
                numeric_i.append(answer_num)

            # Store all numeric scores for this question
            synthetic_answers_clean[question_key] = numeric_i
        
        # ========================================================================
        # SAVE CLEANED RESULTS
        # ========================================================================
        # Save the cleaned results (numeric scores) to JSON file
        with open(clean_file_path, 'w') as f:
            json.dump(synthetic_answers_clean, f)

    # ========================================================================
    # EXECUTE POST-PROCESSING
    # ========================================================================
    # Call the appropriate post-processing function based on dataset type
    if dataset_name == 'EEDI':
        # EEDI: Calculate binary correctness scores (0 or 1)
        process_responses_EEDI(synthetic_answers_raw)
    elif dataset_name == 'OpinionQA':
        # OpinionQA: Calculate numeric opinion scores (in {-1, -1/3, 0, 1/3, 1})
        process_responses_OpinionQA(synthetic_answers_raw)
    
    # ========================================================================
    # ADD A RANDOM BASELINE TO CLEANED RESULTS
    # ========================================================================
    # Generate a random baseline for comparison with synthetic LLM responses.
    # The baseline uses the same format as cleaned responses but with randomly
    # generated values. The size is 2 * num_of_synthetic_answers to provide
    # a sufficient sample size for statistical comparison.
    random_baseline = {}
    for question_key in synthetic_answers_raw.keys():
        # Generate random answers matching the cleaned response format:
        # - EEDI: binary correctness scores (0 = incorrect, 1 = correct; correct with probability 0.25 because there are 4 choices)
        # - OpinionQA: numeric opinion scores (-1 = strongly negative, -1/3 = somewhat negative,
        #   0 = neutral/refused, 1/3 = somewhat positive, 1 = strongly positive; each score with probability 0.2 because there are 5 choices)
        if dataset_name == 'EEDI':
            random_baseline[question_key] = random.choices([0, 1], weights=[0.75, 0.25], k=2 * num_of_synthetic_answers)
        elif dataset_name == 'OpinionQA':
            random_baseline[question_key] = random.choices([-1, -1/3, 0, 1/3, 1], weights=[0.2, 0.2, 0.2, 0.2, 0.2], k=2 * num_of_synthetic_answers)

    # Save the random baseline to JSON file
    with open(random_file_path, 'w') as f:
        json.dump(random_baseline, f)
    
    print(f'Post-processing completed. Results saved to {clean_file_path} and {random_file_path}.')
    
    # ========================================================================
    # CALCULATE AND DISPLAY TOTAL TIME
    # ========================================================================
    # Calculate elapsed time and display in hours
    end_time = time.time()
    elapsed_hours = (end_time - start_time) / 3600
    print(f'Total time taken: {elapsed_hours:.4f} hours.')



